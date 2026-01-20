# -*- coding: utf-8 -*-
"""
工作流控制器

负责工作流的执行控制、断点处理和状态管理。
"""

import logging
from typing import Optional

from PyQt6.QtCore import QObject, pyqtSignal

from src.core.event_bus import get_event_bus
from src.workflow.engine import WorkflowEngine, ExecutionResult
from src.workflow.workflow import Workflow


logger = logging.getLogger(__name__)


class WorkflowController(QObject):
    """
    工作流控制器
    
    职责:
    - 管理工作流的执行生命周期
    - 处理断点暂停和继续
    - 协调引擎事件与视图更新
    
    使用依赖注入模式，接收 WorkflowEngine 实例。
    """
    
    # 控制器信号（供视图订阅）
    execution_started = pyqtSignal()
    execution_finished = pyqtSignal(bool, str)       # success, message
    node_state_changed = pyqtSignal(str, str)        # node_id, state
    breakpoint_triggered = pyqtSignal(str)           # node_id
    
    def __init__(self, engine: WorkflowEngine = None, parent=None):
        """
        初始化工作流控制器
        
        Args:
            engine: 工作流执行引擎（可选，默认创建新实例）
            parent: 父QObject
        """
        super().__init__(parent)
        self._engine = engine or WorkflowEngine()
        self._workflow: Optional[Workflow] = None
        self._event_bus = get_event_bus()
        
        self._connect_engine_signals()
        self._connect_event_bus()
    
    def _connect_engine_signals(self):
        """连接引擎信号"""
        self._engine.workflow_started.connect(self._on_engine_started)
        self._engine.workflow_finished.connect(self._on_engine_finished)
        self._engine.workflow_error.connect(self._on_engine_error)
        self._engine.node_started.connect(self._on_node_started)
        self._engine.node_finished.connect(self._on_node_finished)
        self._engine.node_progress.connect(self._on_node_progress)
        self._engine.breakpoint_hit.connect(self._on_breakpoint_hit)
        self._engine.status_message.connect(self._on_status_message)
    
    def _connect_event_bus(self):
        """连接事件总线信号"""
        self._event_bus.workflow_run_requested.connect(self.run)
        self._event_bus.breakpoint_continue.connect(self.continue_from_breakpoint)
    
    @property
    def engine(self) -> WorkflowEngine:
        """获取执行引擎"""
        return self._engine
    
    @property
    def workflow(self) -> Optional[Workflow]:
        """获取当前工作流"""
        return self._workflow
    
    def set_workflow(self, workflow: Workflow):
        """
        设置当前工作流
        
        Args:
            workflow: 工作流对象
        """
        self._workflow = workflow
        self._engine.set_workflow(workflow)
    
    def run(self) -> bool:
        """
        运行当前工作流
        
        Returns:
            是否成功启动执行
        """
        if not self._workflow:
            self._event_bus.emit_status("没有可运行的工作流")
            return False
        
        # 验证工作流
        valid, errors = self._workflow.validate()
        if not valid:
            error_msg = "工作流验证失败:\n" + "\n".join(errors)
            self._event_bus.workflow_error.emit(error_msg)
            return False
        
        # 开始执行
        self._event_bus.emit_status("正在启动工作流...")
        success = self._engine.execute()
        
        if not success:
            self._event_bus.emit_status("工作流启动失败")
        
        return success
    
    def stop(self):
        """停止工作流执行"""
        self._engine.stop()
        self._event_bus.emit_status("工作流已停止")
    
    def pause(self):
        """暂停工作流执行"""
        self._engine.pause()
        self._event_bus.emit_status("工作流已暂停")
    
    def resume(self):
        """恢复工作流执行"""
        self._engine.resume()
        self._event_bus.emit_status("工作流已恢复")
    
    def continue_from_breakpoint(self):
        """从断点继续执行"""
        if self._engine.is_waiting_at_breakpoint():
            self._engine.continue_from_breakpoint()
            self._event_bus.emit_status("从断点继续执行...")
    
    def is_running(self) -> bool:
        """检查工作流是否正在执行"""
        from src.workflow.engine import EngineState
        return self._engine.state == EngineState.RUNNING
    
    def is_paused(self) -> bool:
        """检查工作流是否已暂停"""
        from src.workflow.engine import EngineState
        return self._engine.state == EngineState.PAUSED
    
    def is_at_breakpoint(self) -> bool:
        """检查是否在断点处等待"""
        return self._engine.is_waiting_at_breakpoint()
    
    def get_current_breakpoint_node(self) -> Optional[str]:
        """获取当前断点节点ID"""
        return self._engine.get_current_breakpoint_node()
    
    # ===== 引擎事件处理 =====
    
    def _on_engine_started(self):
        """引擎开始执行"""
        self.execution_started.emit()
        self._event_bus.workflow_started.emit()
        self._event_bus.emit_status("工作流执行中...")
    
    def _on_engine_finished(self, result: ExecutionResult):
        """引擎执行完成"""
        self.execution_finished.emit(result.success, result.message)
        self._event_bus.workflow_finished.emit(result.success, result.message)
        
        if result.success:
            self._event_bus.emit_status(f"工作流执行完成 (耗时: {result.execution_time:.2f}秒)")
        else:
            self._event_bus.emit_status(f"执行失败: {result.message}")
    
    def _on_engine_error(self, error_msg: str):
        """引擎执行错误"""
        self.execution_finished.emit(False, error_msg)
        self._event_bus.workflow_error.emit(error_msg)
        self._event_bus.emit_status(f"错误: {error_msg}")
    
    def _on_node_started(self, node_id: str):
        """节点开始执行"""
        self.node_state_changed.emit(node_id, "running")
        self._event_bus.node_started.emit(node_id)
        
        if self._workflow:
            node = self._workflow.get_node(node_id)
            if node:
                self._event_bus.emit_status(f"执行: {node.display_name}")
    
    def _on_node_finished(self, node_id: str, success: bool):
        """节点执行完成"""
        state = "completed" if success else "error"
        self.node_state_changed.emit(node_id, state)
        self._event_bus.node_finished.emit(node_id, success)
    
    def _on_node_progress(self, node_id: str, progress: float, data_json: str):
        """节点进度更新"""
        import json
        try:
            data = json.loads(data_json) if data_json else {}
        except json.JSONDecodeError:
            data = {}
        
        self._event_bus.node_progress.emit(node_id, progress, data)
    
    def _on_breakpoint_hit(self, node_id: str):
        """断点触发"""
        self.breakpoint_triggered.emit(node_id)
        self._event_bus.breakpoint_hit.emit(node_id)
        
        if self._workflow:
            node = self._workflow.get_node(node_id)
            node_name = node.display_name if node else node_id
            self._event_bus.emit_status(f"🔴 断点暂停: {node_name}")
    
    def _on_status_message(self, message: str):
        """状态消息"""
        self._event_bus.emit_status(message)


