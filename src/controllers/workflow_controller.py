# -*- coding: utf-8 -*-
"""
Workflow Controller

Responsible for workflow execution control, breakpoint handling, and state management.
"""

import logging
from typing import Optional

from PySide6.QtCore import QObject, Signal

from src.core.event_bus import get_event_bus
from src.ui.i18n import tr_
from src.workflow.engine import WorkflowEngine, ExecutionResult, EngineState
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
    execution_started = Signal()
    execution_finished = Signal(bool, str)       # success, message
    node_state_changed = Signal(str, str)        # node_id, state
    breakpoint_triggered = Signal(str)           # node_id
    
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
        if self._engine.is_active() or self._engine.is_waiting_at_breakpoint():
            self._event_bus.emit_status(tr_("Workflow is active; switching workflow is not allowed"))
            return

        self._workflow = workflow
        self._engine.set_workflow(workflow)

    def validate_workflow(self, workflow: Workflow) -> tuple[bool, list[str]]:
        """
        Validate a workflow and return raw errors without side effects.

        This API is intended for UI to present validation results (e.g. warning dialog)
        without coupling UI to domain validation details.
        """
        try:
            valid, errors = workflow.validate()
            return bool(valid), list(errors or [])
        except Exception as e:
            # Keep a stable return shape; callers decide how to present the error.
            return False, [str(e)]
    
    def run(self, *, validate: bool = True) -> bool:
        """
        运行当前工作流
        
        Returns:
            是否成功启动执行
        """
        if not self._workflow:
            self._event_bus.emit_status(tr_("No workflow to run"))
            return False
        
        if validate:
            valid, errors = self.validate_workflow(self._workflow)
            if not valid:
                error_msg = tr_("Workflow validation failed:\n{errors}").format(
                    errors="\n".join(errors)
                )
                self._event_bus.workflow_error.emit(error_msg)
                return False
        
        # 开始执行
        self._event_bus.emit_status(tr_("Starting workflow..."))
        success = self._engine.execute()
        
        if not success:
            self._event_bus.emit_status(tr_("Failed to start workflow"))
        
        return success

    def run_from_node(self, node_id: str, *, validate: bool = True) -> bool:
        """
        Run the current workflow starting from a selected node.

        Preserves valid upstream cached outputs and reruns only the selected node
        together with its downstream nodes.
        """
        if not self._workflow:
            self._event_bus.emit_status(tr_("No workflow to run"))
            return False

        if validate:
            valid, errors = self.validate_workflow(self._workflow)
            if not valid:
                error_msg = tr_("Workflow validation failed:\n{errors}").format(
                    errors="\n".join(errors)
                )
                self._event_bus.workflow_error.emit(error_msg)
                return False

        self._event_bus.emit_status(tr_("Starting workflow from selected node..."))
        success = self._engine.execute_from(node_id)
        if not success:
            self._event_bus.emit_status(tr_("Failed to start workflow from selected node"))

        return success
    
    def stop(self):
        """停止工作流执行"""
        self._engine.stop()
        # Stopping can be asynchronous (e.g., training node stops at batch/epoch boundary).
        self._event_bus.emit_status(
            tr_("Stop requested. Waiting for current task to finish...")
        )
    
    def pause(self):
        """暂停工作流执行"""
        self._engine.pause()
        self._event_bus.emit_status(tr_("Workflow paused"))
    
    def resume(self):
        """恢复工作流执行"""
        self._engine.resume()
        self._event_bus.emit_status(tr_("Workflow resumed"))
    
    def continue_from_breakpoint(self):
        """从断点继续执行"""
        if self._engine.is_waiting_at_breakpoint():
            self._engine.continue_from_breakpoint()
            self._event_bus.emit_status(tr_("Continuing from breakpoint..."))
    
    def is_running(self) -> bool:
        """检查工作流是否正在执行"""
        return self._engine.is_active()
    
    def is_paused(self) -> bool:
        """检查工作流是否已暂停"""
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
        self._event_bus.emit_status(tr_("Workflow running..."))
    
    def _on_engine_finished(self, result: ExecutionResult):
        """引擎执行完成"""
        self.execution_finished.emit(result.success, result.message)
        self._event_bus.workflow_finished.emit(result.success, result.message)
        
        if result.success:
            self._event_bus.emit_status(
                tr_("Workflow finished (elapsed: {seconds:.2f}s)").format(
                    seconds=result.execution_time
                )
            )
        else:
            self._event_bus.emit_status(
                tr_("Execution failed: {message}").format(message=result.message)
            )
    
    def _on_engine_error(self, error_msg: str):
        """引擎执行错误"""
        self.execution_finished.emit(False, error_msg)
        self._event_bus.workflow_error.emit(error_msg)
        self._event_bus.emit_status(tr_("Error: {message}").format(message=error_msg))
    
    def _on_node_started(self, node_id: str):
        """节点开始执行"""
        self.node_state_changed.emit(node_id, "running")
        self._event_bus.node_started.emit(node_id)
        
        if self._workflow:
            node = self._workflow.get_node(node_id)
            if node:
                self._event_bus.emit_status(
                    tr_("Running: {name}").format(name=node.display_name)
                )
    
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
            self._event_bus.emit_status(
                tr_("🔴 Breakpoint hit: {name}").format(name=node_name)
            )
    
    def _on_status_message(self, message: str):
        """状态消息"""
        self._event_bus.emit_status(message)


