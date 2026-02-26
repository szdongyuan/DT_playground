# -*- coding: utf-8 -*-
"""
Workflow Execution Engine

Responsible for executing workflows, managing data flow and node states.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from src.ui.i18n import tr_
from .connection import Connection
from .node_base import BaseNode, NodeState
from .workflow import Workflow


logger = logging.getLogger(__name__)


class EngineState(Enum):
    """Engine state"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class ExecutionResult:
    """Execution result"""
    success: bool
    message: str
    execution_time: float
    node_results: Dict[str, Any]  # node_id -> output data


class WorkflowEngine(QObject):
    """
    Workflow Execution Engine
    
    Responsible for executing nodes in topological order,
    managing data transfer between nodes.
    
    Signals:
        workflow_started: Workflow started execution
        workflow_finished: Workflow execution completed (ExecutionResult)
        workflow_error: Workflow execution error (error_message)
        node_started: Node started execution (node_id)
        node_finished: Node execution completed (node_id, success)
        node_progress: Node execution progress (node_id, progress, message)
        progress_updated: Overall progress update (current, total, message)
    """
    
    # Signal definitions
    workflow_started = pyqtSignal()
    workflow_finished = pyqtSignal(object)  # ExecutionResult
    workflow_error = pyqtSignal(str)
    node_started = pyqtSignal(str)
    node_finished = pyqtSignal(str, bool)
    node_progress = pyqtSignal(str, float, str)
    progress_updated = pyqtSignal(int, int, str)
    status_message = pyqtSignal(str)  # Status message (displayed in status bar)
    breakpoint_hit = pyqtSignal(str)  # Breakpoint triggered signal (node_id)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.workflow: Optional[Workflow] = None
        self.state = EngineState.IDLE
        
        # Execution control
        self._stop_requested = False
        self._pause_requested = False
        
        # Breakpoint control
        self._breakpoint_waiting = False  # Whether waiting at breakpoint
        self._breakpoint_continue = False  # Whether user confirmed to continue
        self._current_breakpoint_node: Optional[str] = None  # Current breakpoint node ID
        
        # Loop node state
        self._loop_counters: Dict[str, int] = {}
        self._loop_data: Dict[str, List[Any]] = {}
        
        # Worker thread
        self._worker: Optional[EngineWorker] = None
    
    def set_workflow(self, workflow: Workflow):
        """Set workflow to execute"""
        self.workflow = workflow
    
    def execute(self, workflow: Workflow = None) -> bool:
        """
        Execute workflow
        
        Args:
            workflow: Workflow to execute, uses previously set workflow if None
        
        Returns:
            Whether execution started successfully
        """
        if workflow:
            self.workflow = workflow
        
        if not self.workflow:
            self.workflow_error.emit(tr_("Workflow not set"))
            return False

        # Prevent starting a new run while the previous worker thread is still finishing.
        # A terminal state (COMPLETED/STOPPED/ERROR) can be observed before QThread fully exits.
        if self._worker is not None and self._worker.isRunning():
            self.workflow_error.emit(tr_("Workflow is already running"))
            return False
        
        if self.state == EngineState.RUNNING:
            self.workflow_error.emit(tr_("Workflow is already running"))
            return False
        
        # Validate workflow
        valid, errors = self.workflow.validate()
        if not valid:
            self.workflow_error.emit("\n".join(errors))
            return False
        
        # Reset state
        self._reset()
        
        # Create and start worker thread
        self._worker = EngineWorker(self)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()
        
        return True
    
    def execute_sync(self) -> ExecutionResult:
        """
        Execute workflow synchronously (blocks current thread)
        
        Returns:
            Execution result
        """
        if not self.workflow:
            return ExecutionResult(
                success=False,
                message=tr_("Workflow not set"),
                execution_time=0,
                node_results={}
            )
        
        # Validate workflow
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
        """Stop execution"""
        self._stop_requested = True
        self.state = EngineState.STOPPED
        logger.info("Workflow execution stopped")
    
    def pause(self):
        """Pause execution"""
        self._pause_requested = True
        self.state = EngineState.PAUSED
        logger.info("Workflow execution paused")
    
    def resume(self):
        """Resume execution"""
        self._pause_requested = False
        self.state = EngineState.RUNNING
        logger.info("Workflow execution resumed")
    
    def continue_from_breakpoint(self):
        """
        Continue execution from breakpoint
        
        Called by user after previewing breakpoint data to continue workflow execution.
        """
        if self._breakpoint_waiting:
            self._breakpoint_continue = True
            self._breakpoint_waiting = False
            self._current_breakpoint_node = None
            logger.info("User confirmed continue from breakpoint")
    
    def is_waiting_at_breakpoint(self) -> bool:
        """Check if waiting at breakpoint"""
        return self._breakpoint_waiting
    
    def get_current_breakpoint_node(self) -> Optional[str]:
        """Get current breakpoint node ID"""
        return self._current_breakpoint_node
    
    def _reset(self):
        """Reset engine state"""
        self._stop_requested = False
        self._pause_requested = False
        self._breakpoint_waiting = False
        self._breakpoint_continue = False
        self._current_breakpoint_node = None
        self._loop_counters.clear()
        self._loop_data.clear()
        self.state = EngineState.IDLE
        
        # Reset all nodes
        if self.workflow:
            for node in self.workflow.nodes.values():
                node.reset()
    
    def _execute_workflow(self) -> ExecutionResult:
        """
        Core workflow execution logic
        
        Returns:
            Execution result
        """
        start_time = time.time()
        node_results = {}
        
        try:
            self.state = EngineState.RUNNING
            self.workflow_started.emit()
            
            # Get execution order
            execution_order, has_cycle = self.workflow.get_execution_order()
            
            if has_cycle:
                # Has cycle, use special handling
                # Currently simple handling: ignore cycle, execute in topological order
                logger.warning("Detected cyclic dependency, will try to execute in topological order")
            
            total_nodes = len(execution_order)
            
            for i, node_id in enumerate(execution_order):
                # Check stop request
                if self._stop_requested:
                    self.state = EngineState.STOPPED
                    result = ExecutionResult(
                        success=False,
                        message=tr_("Execution stopped"),
                        execution_time=time.time() - start_time,
                        node_results=node_results
                    )
                    self.workflow_finished.emit(result)
                    return result
                
                # Check pause request
                while self._pause_requested:
                    time.sleep(0.1)
                    if self._stop_requested:
                        break
                
                node = self.workflow.get_node(node_id)
                if not node:
                    continue
                
                # Emit progress signal
                self.progress_updated.emit(
                    i + 1,
                    total_nodes,
                    tr_("Running: {name}").format(name=node.display_name),
                )
                
                # Execute node
                success = self._execute_node(node)
                
                if not success:
                    self.state = EngineState.ERROR
                    result = ExecutionResult(
                        success=False,
                        message=tr_("Node '{name}' failed: {error}").format(
                            name=node.display_name,
                            error=node.error_message,
                        ),
                        execution_time=time.time() - start_time,
                        node_results=node_results
                    )
                    self.workflow_finished.emit(result)
                    return result
                
                # Collect results
                node_results[node_id] = {
                    port_name: port.data 
                    for port_name, port in node.outputs.items()
                }
            
            self.state = EngineState.COMPLETED
            result = ExecutionResult(
                success=True,
                message=tr_("Workflow finished"),
                execution_time=time.time() - start_time,
                node_results=node_results
            )
            self.workflow_finished.emit(result)
            return result
            
        except Exception as e:
            logger.exception("Workflow execution exception")
            self.state = EngineState.ERROR
            error_msg = tr_("Execution error: {error}").format(error=str(e))
            self.workflow_error.emit(error_msg)
            return ExecutionResult(
                success=False,
                message=error_msg,
                execution_time=time.time() - start_time,
                node_results=node_results
            )
    
    def _execute_node(self, node: BaseNode) -> bool:
        """
        Execute single node
        
        Args:
            node: Node to execute
        
        Returns:
            Whether execution was successful
        """
        try:
            self.node_started.emit(node.node_id)
            node.state = NodeState.RUNNING
            
            # Collect input data
            self._collect_node_inputs(node)
            
            # Validate node
            valid, msg = node.validate()
            if not valid:
                node.state = NodeState.ERROR
                node.error_message = msg
                self.node_finished.emit(node.node_id, False)
                return False
            
            # Set progress callback
            import json
            def make_progress_callback(node_id):
                def callback(progress, message, data):
                    # Convert data dict to JSON string for transmission
                    data_str = json.dumps(data) if data else "{}"
                    self.node_progress.emit(node_id, progress, data_str)
                return callback
            
            node.progress_callback = make_progress_callback(node.node_id)
            
            # Set status message callback
            node.status_callback = lambda msg: self.status_message.emit(msg)
            
            # Execute node
            logger.debug(f"Executing node: {node.display_name} ({node.node_id})")
            success = node.execute()
            
            if success:
                node.state = NodeState.COMPLETED
                # Propagate output data
                self._propagate_node_outputs(node)
                
                # Check if breakpoint node
                if self._check_and_handle_breakpoint(node):
                    # Breakpoint handling, waiting for user to continue
                    pass
            else:
                node.state = NodeState.ERROR
            
            self.node_finished.emit(node.node_id, success)
            return success
            
        except Exception as e:
            logger.exception(f"Node execution exception: {node.node_id}")
            node.state = NodeState.ERROR
            node.error_message = str(e)
            self.node_finished.emit(node.node_id, False)
            return False
    
    def _check_and_handle_breakpoint(self, node: BaseNode) -> bool:
        """
        Check if node is a breakpoint node, if so handle breakpoint pause
        
        Args:
            node: Just executed node
        
        Returns:
            Whether is breakpoint node and paused
        """
        # Check if is breakpoint node
        if not getattr(node, 'is_breakpoint', False):
            return False
        
        # Check if breakpoint is enabled
        if hasattr(node, 'is_breakpoint_enabled') and not node.is_breakpoint_enabled():
            logger.debug(f"Breakpoint node {node.node_id} is disabled, skipping")
            return False
        
        # Trigger breakpoint
        logger.info(f"Breakpoint triggered: {node.display_name} ({node.node_id})")
        self._breakpoint_waiting = True
        self._breakpoint_continue = False
        self._current_breakpoint_node = node.node_id
        self.state = EngineState.PAUSED
        
        # Emit breakpoint signal
        self.breakpoint_hit.emit(node.node_id)
        self.status_message.emit(
            tr_("Breakpoint paused: {name} - click Continue to proceed").format(
                name=node.display_name
            )
        )
        
        # Wait for user to continue (blocks current thread)
        while self._breakpoint_waiting and not self._stop_requested:
            time.sleep(0.1)
        
        # Check if stopped
        if self._stop_requested:
            return True
        
        # User confirmed continue
        self.state = EngineState.RUNNING
        self.status_message.emit(tr_("Continuing from breakpoint..."))
        return True
    
    def _collect_node_inputs(self, node: BaseNode):
        """Collect node input data (from upstream node outputs)"""
        for conn in self.workflow.get_incoming_connections(node.node_id):
            source_node = self.workflow.get_node(conn.source_node_id)
            if source_node and conn.source_port in source_node.outputs:
                source_data = source_node.outputs[conn.source_port].data
                if conn.target_port in node.inputs:
                    node.inputs[conn.target_port].data = source_data
    
    def _propagate_node_outputs(self, node: BaseNode):
        """Propagate node output data to downstream nodes"""
        # Data is already stored in output ports, downstream nodes will get it via _collect_node_inputs
        pass
    
    def _on_worker_finished(self):
        """Worker thread completion callback"""
        self._worker = None
        # Once the worker thread exits, the engine is ready for the next execution.
        if self.state in (EngineState.COMPLETED, EngineState.ERROR, EngineState.STOPPED):
            self.state = EngineState.IDLE
    
    def get_node_output(self, node_id: str, port_name: str = None) -> Any:
        """
        Get node output data
        
        Args:
            node_id: Node ID
            port_name: Port name, returns first output port data if None
        
        Returns:
            Output data
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
            # Return first output port data
            if node.outputs:
                first_port = next(iter(node.outputs.values()))
                return first_port.data
        
        return None


class EngineWorker(QThread):
    """Workflow execution thread"""
    
    def __init__(self, engine: WorkflowEngine):
        super().__init__()
        self.engine = engine
    
    def run(self):
        """Thread execution"""
        self.engine._execute_workflow()

