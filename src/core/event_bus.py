# -*- coding: utf-8 -*-
"""
Global Event Bus

Uses mediator pattern to decouple communication between views.
"""

import logging
from typing import Any, Dict, Optional

from PySide6.QtCore import QObject, Signal


logger = logging.getLogger(__name__)


class EventBus(QObject):
    """
    Global Event Bus
    
    Uses mediator pattern to centrally manage event communication within the application,
    decoupling direct dependencies between views, controllers, and services.
    
    Usage Guidelines
    ----------------
    1. **Cross-component communication**: Use EventBus
       - Communication between different views (e.g., WorkflowView -> TrainingView)
       - Broadcasting from Controller to multiple Views
       - Global state change notifications
       
    2. **Intra-component communication**: Use Signal
       - Communication between child components within the same class
       - Direct communication between parent and child components
       - Communication between Widget and its internal elements
    
    Examples
    --------
    Cross-component communication (recommended to use EventBus)::
    
        # Emit event in WorkflowController
        self._event_bus.workflow_started.emit()
        
        # Subscribe to event in TrainingView
        get_event_bus().workflow_started.connect(self._on_workflow_started)
    
    Intra-component communication (recommended to use Signal)::
    
        class WorkflowView(QWidget):
            # Component signals for parent to subscribe
            workflow_changed = Signal()
            node_selected = Signal(str)
            
            def _on_internal_change(self):
                self.workflow_changed.emit()  # Use component signal directly
    """
    
    _instance: Optional['EventBus'] = None
    
    # ===== Workflow Events =====
    workflow_run_requested = Signal()                       # Request to run workflow
    workflow_started = Signal()                             # Workflow started execution
    workflow_finished = Signal(bool, str)                   # Execution finished (success, message)
    workflow_error = Signal(str)                            # Execution error (error_message)
    workflow_saved = Signal(str)                            # Workflow saved (filepath)
    workflow_loaded = Signal(str)                           # Workflow loaded (filepath)
    
    # ===== Node Events =====
    node_selected = Signal(str)                             # Node selected (node_id)
    node_deselected = Signal()                              # Selection cleared
    node_started = Signal(str)                              # Node started execution (node_id)
    node_finished = Signal(str, bool)                       # Node finished execution (node_id, success)
    node_progress = Signal(str, float, dict)                # Node progress (node_id, progress, data)
    
    # ===== Breakpoint Events =====
    breakpoint_hit = Signal(str)                            # Breakpoint triggered (node_id)
    breakpoint_continue = Signal()                          # Continue execution
    
    # ===== Preview Events =====
    preview_requested = Signal(str, dict)                   # Preview requested (node_id, outputs)
    preview_updated = Signal(str, str, dict)                # Preview updated (node_id, node_name, outputs)
    
    # ===== Training Events =====
    training_started = Signal(int)                          # Training started (total_epochs)
    training_epoch_completed = Signal(int, int, dict)       # Epoch completed (current, total, metrics)
    training_finished = Signal(bool, str)                   # Training finished (success, message)
    training_stopped = Signal()                             # Training stopped
    training_stop_with_checkpoint = Signal(str)             # Request stop + save checkpoint (path)
    
    # ===== Model Events =====
    model_loaded = Signal(object)                           # Model loaded (model)
    model_saved = Signal(str)                               # Model saved (filepath)
    model_built = Signal(object)                            # Model built (model)
    
    # ===== View Navigation Events =====
    view_switch_requested = Signal(int)                     # View switch requested (view_index)
    
    # ===== Status Message Events =====
    status_message = Signal(str)                            # Status bar message
    status_message_temporary = Signal(str, int)             # Temporary message (message, duration_ms)
    
    def __init__(self, parent=None):
        super().__init__(parent)
    
    @classmethod
    def instance(cls) -> 'EventBus':
        """
        Get global singleton instance
        
        Returns:
            EventBus instance
        """
        if cls._instance is None:
            cls._instance = cls()
            logger.debug("EventBus instance created")
        return cls._instance
    
    @classmethod
    def reset(cls):
        """
        Reset singleton instance (for testing only)
        """
        if cls._instance is not None:
            cls._instance.deleteLater()
            cls._instance = None
            logger.debug("EventBus instance reset")
    
    def emit_status(self, message: str):
        """
        Convenience method to emit status message
        
        Args:
            message: Status message
        """
        self.status_message.emit(message)
    
    def emit_node_progress(self, node_id: str, progress: float, **kwargs):
        """
        Convenience method to emit node progress
        
        Args:
            node_id: Node ID
            progress: Progress value (0.0 ~ 1.0)
            **kwargs: Additional data
        """
        self.node_progress.emit(node_id, progress, kwargs)
    
    def emit_training_epoch(self, current: int, total: int, **metrics):
        """
        Convenience method to emit training epoch completion
        
        Args:
            current: Current epoch
            total: Total epochs
            **metrics: Training metrics (loss, accuracy, val_loss, val_accuracy, etc.)
        """
        self.training_epoch_completed.emit(current, total, metrics)


def get_event_bus() -> EventBus:
    """
    Get global event bus instance
    
    This is a convenience function for EventBus.instance().
    
    Returns:
        EventBus instance
    """
    return EventBus.instance()


