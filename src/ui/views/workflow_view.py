# -*- coding: utf-8 -*-
"""
Workflow View

Thin wrapper that composes WorkflowToolbar + WorkflowEditorWidget.
Retains backward-compatible API used by legacy callsites.
"""

import logging
import os
from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from ..i18n import tr_
from ...controllers.workflow_controller import WorkflowController
from ...workflow.engine import WorkflowEngine
from ...workflow.workflow import Workflow
from src.utils.config import config

from .workflow_editor_widget import WorkflowEditorWidget
from .workflow_toolbar import WorkflowToolbar


logger = logging.getLogger(__name__)


class WorkflowView(QWidget):
    """
    Workflow editor view (backward-compatible wrapper).

    Composes a WorkflowToolbar and a WorkflowEditorWidget.
    When *show_toolbar* is False only the editor is shown (useful for embedding).
    """

    workflow_changed = Signal()
    run_requested = Signal()
    node_selected = Signal(str)
    node_double_clicked = Signal(str)
    run_from_node_requested = Signal(str)
    continue_requested = Signal()

    def __init__(self, parent=None, *, show_toolbar: bool = True):
        super().__init__(parent)

        self._workflow: Optional[Workflow] = None
        self._engine: Optional[WorkflowEngine] = None
        self._workflow_controller: Optional[WorkflowController] = None

        self._setup_ui(show_toolbar)
        self._connect_signals()

    # ===== UI =====

    def _setup_ui(self, show_toolbar: bool):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._toolbar: Optional[WorkflowToolbar] = None
        if show_toolbar:
            self._toolbar = WorkflowToolbar(show_duplicate=False)
            layout.addWidget(self._toolbar)

        self._editor = WorkflowEditorWidget()
        layout.addWidget(self._editor)

    def _connect_signals(self):
        self._editor.workflow_changed.connect(self._on_workflow_changed)
        self._editor.node_selected.connect(self.node_selected)
        self._editor.node_double_clicked.connect(self.node_double_clicked)
        self._editor.run_from_node_requested.connect(self.run_from_node_requested)

        if self._toolbar is not None:
            self._toolbar.new_requested.connect(self._on_new_workflow)
            self._toolbar.open_requested.connect(self._on_open_workflow)
            self._toolbar.save_requested.connect(self._on_save_workflow)
            self._toolbar.run_requested.connect(self.run_requested)
            self._toolbar.stop_requested.connect(self._on_stop_workflow)
            self._toolbar.continue_requested.connect(self.continue_requested)
            self._toolbar.fit_requested.connect(self._editor.fit_to_selection)
            self._toolbar.clear_requested.connect(self._on_clear_workflow)

    # ===== Public API (editor delegation) =====

    def set_workflow(self, workflow: Workflow):
        self._workflow = workflow
        self._editor.set_workflow(workflow)
        self._update_toolbar_title()

    def get_workflow(self) -> Optional[Workflow]:
        return self._editor.get_workflow()

    def update_node_state(self, node_id: str, state: str):
        self._editor.update_node_state(node_id, state)

    def reset_all_node_states(self):
        self._editor.reset_all_node_states()

    def reset_node_states(self, node_ids: list[str]):
        self._editor.reset_node_states(node_ids)

    def highlight_node(self, node_id: str):
        self._editor.highlight_node(node_id)

    def fit_to_selection(self):
        self._editor.fit_to_selection()

    # ===== Public API (toolbar delegation) =====

    def set_run_controls_state(
        self,
        *,
        run_enabled: bool,
        stop_enabled: bool,
        stop_text: Optional[str] = None,
    ) -> None:
        if self._toolbar is not None:
            self._toolbar.set_run_controls_state(
                run_enabled=run_enabled,
                stop_enabled=stop_enabled,
                stop_text=stop_text,
            )

    def show_breakpoint_mode(self, enabled: bool, node_name: str = ""):
        if self._toolbar is not None:
            self._toolbar.show_breakpoint_mode(enabled, node_name)

    # ===== Backward-compat setters =====

    def set_engine(self, engine: WorkflowEngine):
        self._engine = engine

    def set_workflow_controller(self, controller: WorkflowController):
        self._workflow_controller = controller
        self._engine = controller.engine

    # ===== Business logic (New / Open / Save / Clear) =====

    def _on_new_workflow(self):
        self._workflow = Workflow("new_workflow")
        self._workflow._file_path = None
        self._editor.set_workflow(self._workflow)
        self._update_toolbar_title()
        self.workflow_changed.emit()

    def _on_open_workflow(self):
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(
            self,
            tr_("Open workflow"),
            "workflows",
            tr_("Workflow files (*.json)"),
        )
        if path:
            workflow = Workflow.load(path)
            if workflow:
                workflow._file_path = path
                self.set_workflow(workflow)
                self.workflow_changed.emit()
                try:
                    config.set("session.last_workflow_path", path)
                    config.add_recent_file(path)
                except Exception:
                    pass

    def _on_save_workflow(self):
        from PySide6.QtWidgets import QFileDialog

        workflow = self._editor.get_workflow()
        if not workflow:
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            tr_("Save workflow"),
            f"workflows/{workflow.name}.json",
            tr_("Workflow files (*.json)"),
        )
        if path:
            workflow.save(path)
            workflow._file_path = path
            self._update_toolbar_title()
            try:
                config.set("session.last_workflow_path", path)
                config.add_recent_file(path)
            except Exception:
                pass

    def _on_stop_workflow(self):
        from src.core.event_bus import get_event_bus

        if self._toolbar is not None:
            self._toolbar.set_run_controls_state(
                run_enabled=False,
                stop_enabled=False,
                stop_text=tr_("⏳ Stopping..."),
            )

        event_bus = get_event_bus()
        event_bus.emit_status(tr_("Stopping workflow... (waiting for current task to finish)"))
        event_bus.training_stopped.emit()

        if self._workflow_controller is not None:
            self._workflow_controller.stop()
        elif self._engine:
            self._engine.stop()

    def _on_clear_workflow(self):
        from PySide6.QtWidgets import QMessageBox

        reply = QMessageBox.question(
            self,
            tr_("Confirm"),
            tr_("Clear the current workflow?"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._on_new_workflow()

    # ===== Internal helpers =====

    def _on_workflow_changed(self):
        self._update_toolbar_title()
        self.workflow_changed.emit()

    def _update_toolbar_title(self):
        if self._toolbar is None:
            return

        workflow = self._workflow
        if workflow and hasattr(workflow, "_file_path") and workflow._file_path:
            filename = os.path.basename(workflow._file_path)
            name = os.path.splitext(filename)[0]
        elif workflow:
            name = workflow.name if workflow.name else "new_workflow"
        else:
            name = "new_workflow"

        self._toolbar.set_title(name)
