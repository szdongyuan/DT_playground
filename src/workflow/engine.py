# -*- coding: utf-8 -*-
"""
工作流执行引擎

负责执行工作流，管理数据流动和节点状态。
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from .connection import Connection
from .node_base import BaseNode, NodeState
from .workflow import Workflow


logger = logging.getLogger(__name__)


class EngineState(Enum):
    """引擎状态"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class ExecutionResult:
    """执行结果"""
    success: bool
    message: str
    execution_time: float
    node_results: Dict[str, Any]  # node_id -> output data


class WorkflowEngine(QObject):
    """
    工作流执行引擎
    
    负责按拓扑顺序执行工作流中的节点，
    管理节点间的数据传递。
    
    Signals:
        workflow_started: 工作流开始执行
        workflow_finished: 工作流执行完成 (ExecutionResult)
        workflow_error: 工作流执行错误 (error_message)
        node_started: 节点开始执行 (node_id)
        node_finished: 节点执行完成 (node_id, success)
        node_progress: 节点执行进度 (node_id, progress, message)
        progress_updated: 整体进度更新 (current, total, message)
    """
    
    # 信号定义
    workflow_started = pyqtSignal()
    workflow_finished = pyqtSignal(object)  # ExecutionResult
    workflow_error = pyqtSignal(str)
    node_started = pyqtSignal(str)
    node_finished = pyqtSignal(str, bool)
    node_progress = pyqtSignal(str, float, str)
    progress_updated = pyqtSignal(int, int, str)
    status_message = pyqtSignal(str)  # 状态消息（显示在状态栏）
    breakpoint_hit = pyqtSignal(str)  # 断点触发信号 (node_id)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.workflow: Optional[Workflow] = None
        self.state = EngineState.IDLE
        
        # 执行控制
        self._stop_requested = False
        self._pause_requested = False
        
        # 断点控制
        self._breakpoint_waiting = False  # 是否在等待断点继续
        self._breakpoint_continue = False  # 用户是否确认继续
        self._current_breakpoint_node: Optional[str] = None  # 当前断点节点ID
        
        # 循环节点状态
        self._loop_counters: Dict[str, int] = {}
        self._loop_data: Dict[str, List[Any]] = {}
        
        # 工作线程
        self._worker: Optional[EngineWorker] = None
    
    def set_workflow(self, workflow: Workflow):
        """设置要执行的工作流"""
        self.workflow = workflow
    
    def execute(self, workflow: Workflow = None) -> bool:
        """
        执行工作流
        
        Args:
            workflow: 要执行的工作流，如果为 None 则使用之前设置的工作流
        
        Returns:
            是否成功启动执行
        """
        if workflow:
            self.workflow = workflow
        
        if not self.workflow:
            self.workflow_error.emit("未设置工作流")
            return False
        
        if self.state == EngineState.RUNNING:
            self.workflow_error.emit("工作流正在执行中")
            return False
        
        # 验证工作流
        valid, errors = self.workflow.validate()
        if not valid:
            self.workflow_error.emit("\n".join(errors))
            return False
        
        # 重置状态
        self._reset()
        
        # 创建并启动工作线程
        self._worker = EngineWorker(self)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()
        
        return True
    
    def execute_sync(self) -> ExecutionResult:
        """
        同步执行工作流（阻塞当前线程）
        
        Returns:
            执行结果
        """
        if not self.workflow:
            return ExecutionResult(
                success=False,
                message="未设置工作流",
                execution_time=0,
                node_results={}
            )
        
        # 验证工作流
        valid, errors = self.workflow.validate()
        if not valid:
            return ExecutionResult(
                success=False,
                message="\n".join(errors),
                execution_time=0,
                node_results={}
            )
        
        self._reset()
        return self._execute_workflow()
    
    def stop(self):
        """停止执行"""
        self._stop_requested = True
        self.state = EngineState.STOPPED
        logger.info("工作流执行已停止")
    
    def pause(self):
        """暂停执行"""
        self._pause_requested = True
        self.state = EngineState.PAUSED
        logger.info("工作流执行已暂停")
    
    def resume(self):
        """恢复执行"""
        self._pause_requested = False
        self.state = EngineState.RUNNING
        logger.info("工作流执行已恢复")
    
    def continue_from_breakpoint(self):
        """
        从断点处继续执行
        
        用户在预览断点数据后调用此方法继续工作流执行。
        """
        if self._breakpoint_waiting:
            self._breakpoint_continue = True
            self._breakpoint_waiting = False
            self._current_breakpoint_node = None
            logger.info("用户确认从断点继续执行")
    
    def is_waiting_at_breakpoint(self) -> bool:
        """检查是否正在断点处等待"""
        return self._breakpoint_waiting
    
    def get_current_breakpoint_node(self) -> Optional[str]:
        """获取当前断点节点ID"""
        return self._current_breakpoint_node
    
    def _reset(self):
        """重置引擎状态"""
        self._stop_requested = False
        self._pause_requested = False
        self._breakpoint_waiting = False
        self._breakpoint_continue = False
        self._current_breakpoint_node = None
        self._loop_counters.clear()
        self._loop_data.clear()
        self.state = EngineState.IDLE
        
        # 重置所有节点
        if self.workflow:
            for node in self.workflow.nodes.values():
                node.reset()
    
    def _execute_workflow(self) -> ExecutionResult:
        """
        执行工作流的核心逻辑
        
        Returns:
            执行结果
        """
        start_time = time.time()
        node_results = {}
        
        try:
            self.state = EngineState.RUNNING
            self.workflow_started.emit()
            
            # 获取执行顺序
            execution_order, has_cycle = self.workflow.get_execution_order()
            
            if has_cycle:
                # 存在循环，使用特殊处理
                # 目前简单处理：忽略循环，按普通顺序执行
                logger.warning("检测到循环依赖，将尝试按拓扑顺序执行")
            
            total_nodes = len(execution_order)
            
            for i, node_id in enumerate(execution_order):
                # 检查停止请求
                if self._stop_requested:
                    return ExecutionResult(
                        success=False,
                        message="执行已停止",
                        execution_time=time.time() - start_time,
                        node_results=node_results
                    )
                
                # 检查暂停请求
                while self._pause_requested:
                    time.sleep(0.1)
                    if self._stop_requested:
                        break
                
                node = self.workflow.get_node(node_id)
                if not node:
                    continue
                
                # 发送进度信号
                self.progress_updated.emit(i + 1, total_nodes, f"执行: {node.display_name}")
                
                # 执行节点
                success = self._execute_node(node)
                
                if not success:
                    self.state = EngineState.ERROR
                    return ExecutionResult(
                        success=False,
                        message=f"节点 '{node.display_name}' 执行失败: {node.error_message}",
                        execution_time=time.time() - start_time,
                        node_results=node_results
                    )
                
                # 收集结果
                node_results[node_id] = {
                    port_name: port.data 
                    for port_name, port in node.outputs.items()
                }
            
            self.state = EngineState.COMPLETED
            result = ExecutionResult(
                success=True,
                message="工作流执行完成",
                execution_time=time.time() - start_time,
                node_results=node_results
            )
            self.workflow_finished.emit(result)
            return result
            
        except Exception as e:
            logger.exception("工作流执行异常")
            self.state = EngineState.ERROR
            error_msg = f"执行异常: {str(e)}"
            self.workflow_error.emit(error_msg)
            return ExecutionResult(
                success=False,
                message=error_msg,
                execution_time=time.time() - start_time,
                node_results=node_results
            )
    
    def _execute_node(self, node: BaseNode) -> bool:
        """
        执行单个节点
        
        Args:
            node: 要执行的节点
        
        Returns:
            是否执行成功
        """
        try:
            self.node_started.emit(node.node_id)
            node.state = NodeState.RUNNING
            
            # 收集输入数据
            self._collect_node_inputs(node)
            
            # 验证节点
            valid, msg = node.validate()
            if not valid:
                node.state = NodeState.ERROR
                node.error_message = msg
                self.node_finished.emit(node.node_id, False)
                return False
            
            # 设置进度回调
            import json
            def make_progress_callback(node_id):
                def callback(progress, message, data):
                    # 将 data 字典转换为 JSON 字符串以便传递
                    data_str = json.dumps(data) if data else "{}"
                    self.node_progress.emit(node_id, progress, data_str)
                return callback
            
            node.progress_callback = make_progress_callback(node.node_id)
            
            # 设置状态消息回调
            node.status_callback = lambda msg: self.status_message.emit(msg)
            
            # 执行节点
            logger.debug(f"执行节点: {node.display_name} ({node.node_id})")
            success = node.execute()
            
            if success:
                node.state = NodeState.COMPLETED
                # 传播输出数据
                self._propagate_node_outputs(node)
                
                # 检查是否是断点节点
                if self._check_and_handle_breakpoint(node):
                    # 断点处理中，等待用户继续
                    pass
            else:
                node.state = NodeState.ERROR
            
            self.node_finished.emit(node.node_id, success)
            return success
            
        except Exception as e:
            logger.exception(f"节点执行异常: {node.node_id}")
            node.state = NodeState.ERROR
            node.error_message = str(e)
            self.node_finished.emit(node.node_id, False)
            return False
    
    def _check_and_handle_breakpoint(self, node: BaseNode) -> bool:
        """
        检查节点是否是断点节点，如果是则处理断点暂停
        
        Args:
            node: 刚执行完的节点
        
        Returns:
            是否是断点节点且已暂停
        """
        # 检查是否是断点节点
        if not getattr(node, 'is_breakpoint', False):
            return False
        
        # 检查断点是否启用
        if hasattr(node, 'is_breakpoint_enabled') and not node.is_breakpoint_enabled():
            logger.debug(f"断点节点 {node.node_id} 已禁用，跳过")
            return False
        
        # 触发断点
        logger.info(f"触发断点: {node.display_name} ({node.node_id})")
        self._breakpoint_waiting = True
        self._breakpoint_continue = False
        self._current_breakpoint_node = node.node_id
        self.state = EngineState.PAUSED
        
        # 发送断点信号
        self.breakpoint_hit.emit(node.node_id)
        self.status_message.emit(f"断点暂停: {node.display_name} - 点击继续按钮以继续执行")
        
        # 等待用户继续（阻塞当前线程）
        while self._breakpoint_waiting and not self._stop_requested:
            time.sleep(0.1)
        
        # 检查是否被停止
        if self._stop_requested:
            return True
        
        # 用户已确认继续
        self.state = EngineState.RUNNING
        self.status_message.emit("从断点继续执行...")
        return True
    
    def _collect_node_inputs(self, node: BaseNode):
        """收集节点的输入数据（来自上游节点的输出）"""
        for conn in self.workflow.get_incoming_connections(node.node_id):
            source_node = self.workflow.get_node(conn.source_node_id)
            if source_node and conn.source_port in source_node.outputs:
                source_data = source_node.outputs[conn.source_port].data
                if conn.target_port in node.inputs:
                    node.inputs[conn.target_port].data = source_data
    
    def _propagate_node_outputs(self, node: BaseNode):
        """将节点的输出数据传播到下游节点"""
        # 数据已经存储在输出端口中，下游节点执行时会通过 _collect_node_inputs 获取
        pass
    
    def _on_worker_finished(self):
        """工作线程完成回调"""
        self._worker = None
    
    def get_node_output(self, node_id: str, port_name: str = None) -> Any:
        """
        获取节点的输出数据
        
        Args:
            node_id: 节点ID
            port_name: 端口名称，如果为 None 则返回第一个输出端口的数据
        
        Returns:
            输出数据
        """
        if not self.workflow:
            return None
        
        node = self.workflow.get_node(node_id)
        if not node:
            return None
        
        if port_name:
            if port_name in node.outputs:
                return node.outputs[port_name].data
        else:
            # 返回第一个输出端口的数据
            if node.outputs:
                first_port = next(iter(node.outputs.values()))
                return first_port.data
        
        return None


class EngineWorker(QThread):
    """工作流执行线程"""
    
    def __init__(self, engine: WorkflowEngine):
        super().__init__()
        self.engine = engine
    
    def run(self):
        """线程执行"""
        self.engine._execute_workflow()

