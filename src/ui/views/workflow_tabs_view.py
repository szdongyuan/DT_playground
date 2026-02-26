"""
Workflow Tabs View

Tab container for multiple workflow editors.
Composes WorkflowToolbar + QTabWidget(WorkflowEditorWidget * N).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QMessageBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.controllers.workflow_controller import WorkflowController
from src.core.event_bus import get_event_bus
from src.ui.i18n import tr_
from src.utils.config import config
from src.workflow.workflow import Workflow

from .workflow_editor_widget import WorkflowEditorWidget
from .workflow_toolbar import WorkflowToolbar


@dataclass
class _Tab:
    editor: WorkflowEditorWidget


class WorkflowTabsView(QWidget):
    """
    Workflow tab container.

    Exposes the same public API surface as the old implementation so that
    MainWindow and App can work without changes.
    """

    workflow_changed = pyqtSignal()
    run_requested = pyqtSignal()
    node_selected = pyqtSignal(str)
    node_double_clicked = pyqtSignal(str)
    continue_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._controller: Optional[WorkflowController] = None

        self._setup_ui()
        self._connect_signals()

        # Always keep at least one tab.
        self.new_tab()

    # ===== Public APIs (MainWindow compatibility) =====

    def set_workflow_controller(self, controller: WorkflowController):
        self._controller = controller

    def set_workflow(self, workflow: Workflow):
        """Backward-compatible: set workflow into *current* editor (no new tab)."""
        editor = self._current_editor()
        if editor is None:
            self.new_tab()
            editor = self._current_editor()
        if editor is None:
            return

        editor.set_workflow(workflow)
        self._refresh_titles()
        self.workflow_changed.emit()

    def get_workflow(self) -> Optional[Workflow]:
        editor = self._current_editor()
        return editor.get_workflow() if editor else None

    def reset_all_node_states(self):
        editor = self._current_editor()
        if editor:
            editor.reset_all_node_states()

    def update_node_state(self, node_id: str, state: str):
        editor = self._current_editor()
        if editor:
            editor.update_node_state(node_id, state)

    def highlight_node(self, node_id: str):
        editor = self._current_editor()
        if editor:
            editor.highlight_node(node_id)

    def show_breakpoint_mode(self, enabled: bool, node_name: str = ""):
        self._toolbar.show_breakpoint_mode(enabled, node_name)

    def set_run_controls_state(
        self,
        *,
        run_enabled: bool,
        stop_enabled: bool,
        stop_text: Optional[str] = None,
    ) -> None:
        self._toolbar.set_run_controls_state(
            run_enabled=run_enabled,
            stop_enabled=stop_enabled,
            stop_text=stop_text,
        )

    # ===== Tab operations =====

    def new_tab(self):
        workflow = Workflow("new_workflow")
        workflow._file_path = None  # type: ignore[attr-defined]
        self._add_editor_tab(workflow=workflow, make_current=True)

    def open_workflow_file(self, path: str):
        workflow = Workflow.load(path)
        if not workflow:
            QMessageBox.warning(
                self,
                tr_("Open workflow"),
                tr_("Failed to load workflow: {path}").format(path=path),
            )
            return

        workflow._file_path = path  # type: ignore[attr-defined]
        self._add_editor_tab(workflow=workflow, make_current=True)

        try:
            config.set("session.last_workflow_path", path)
            config.add_recent_file(path)
        except Exception:
            pass

    def duplicate_current_tab(self):
        workflow = self.get_workflow()
        if not workflow:
            return

        try:
            snapshot = workflow.to_dict()
            cloned = Workflow.from_dict(snapshot)
        except Exception:
            return

        try:
            cloned._file_path = None  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            cloned.name = f"{workflow.name} (copy)"
        except Exception:
            pass
        try:
            cloned._mark_dirty()
        except Exception:
            pass

        self._add_editor_tab(workflow=cloned, make_current=True)

    # ===== UI =====

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._toolbar = WorkflowToolbar(show_duplicate=True)
        layout.addWidget(self._toolbar)

        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        self._tabs.setTabsClosable(True)
        layout.addWidget(self._tabs)

    def _connect_signals(self):
        # Tab widget signals
        self._tabs.tabCloseRequested.connect(self._on_tab_close_requested)
        self._tabs.currentChanged.connect(self._on_current_changed)

        # Toolbar command signals -> container logic
        self._toolbar.new_requested.connect(self.new_tab)
        self._toolbar.open_requested.connect(self._open_dialog)
        self._toolbar.save_requested.connect(self._save_current)
        self._toolbar.duplicate_requested.connect(self.duplicate_current_tab)
        self._toolbar.run_requested.connect(self.run_requested)
        self._toolbar.stop_requested.connect(self._on_stop_workflow)
        self._toolbar.continue_requested.connect(self.continue_requested)
        self._toolbar.fit_requested.connect(self._fit_current)
        self._toolbar.clear_requested.connect(self._clear_current)

    # ===== Internals =====

    def _add_editor_tab(self, *, workflow: Workflow, make_current: bool):
        editor = WorkflowEditorWidget()
        editor.set_workflow(workflow)

        editor.workflow_changed.connect(self._on_editor_workflow_changed)
        editor.node_selected.connect(self.node_selected)
        editor.node_double_clicked.connect(self.node_double_clicked)

        idx = self._tabs.addTab(editor, "")
        self._update_tab_title(idx)
        if make_current:
            self._tabs.setCurrentIndex(idx)
            self._refresh_titles()
            self.workflow_changed.emit()

    def _get_tab(self, index: int) -> Optional[_Tab]:
        w = self._tabs.widget(index)
        if isinstance(w, WorkflowEditorWidget):
            return _Tab(editor=w)
        return None

    def _current_editor(self) -> Optional[WorkflowEditorWidget]:
        tab = self._get_tab(self._tabs.currentIndex())
        return tab.editor if tab else None

    # ===== Title helpers =====

    def _workflow_display_name(self, workflow: Optional[Workflow]) -> str:
        if not workflow:
            return "new_workflow"

        file_path = getattr(workflow, "_file_path", None)
        if file_path:
            filename = os.path.basename(file_path)
            return os.path.splitext(filename)[0]

        return workflow.name or "new_workflow"

    def _tab_title_for_workflow(self, workflow: Optional[Workflow]) -> str:
        name = self._workflow_display_name(workflow)
        if workflow and getattr(workflow, "is_dirty", False):
            return f"{name} *"
        return name

    def _update_tab_title(self, index: int):
        tab = self._get_tab(index)
        workflow = tab.editor.get_workflow() if tab else None
        self._tabs.setTabText(index, self._tab_title_for_workflow(workflow))

    def _refresh_titles(self):
        idx = self._tabs.currentIndex()
        self._update_tab_title(idx)
        workflow = self.get_workflow()
        self._toolbar.set_title(self._tab_title_for_workflow(workflow))

    # ===== Event handlers =====

    def _on_editor_workflow_changed(self):
        self._refresh_titles()
        self.workflow_changed.emit()

    def _on_current_changed(self, index: int):
        self._update_tab_title(index)
        self._refresh_titles()
        self.workflow_changed.emit()

    def _open_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            tr_("Open workflow"),
            "workflows",
            tr_("Workflow files (*.json)"),
        )
        if path:
            self.open_workflow_file(path)

    def _save_current(self):
        editor = self._current_editor()
        if not editor:
            return
        self._save_editor(editor)

    def _save_editor(self, editor: WorkflowEditorWidget) -> bool:
        workflow = editor.get_workflow()
        if not workflow:
            return True

        file_path = getattr(workflow, "_file_path", None)
        if file_path:
            ok = workflow.save(file_path)
            if ok:
                try:
                    config.set("session.last_workflow_path", file_path)
                    config.add_recent_file(file_path)
                except Exception:
                    pass
                self._on_editor_workflow_changed()
            return bool(ok)

        path, _ = QFileDialog.getSaveFileName(
            self,
            tr_("Save workflow"),
            f"workflows/{workflow.name}.json",
            tr_("Workflow files (*.json)"),
        )
        if not path:
            return False

        ok = workflow.save(path)
        if ok:
            workflow._file_path = path  # type: ignore[attr-defined]
            try:
                config.set("session.last_workflow_path", path)
                config.add_recent_file(path)
            except Exception:
                pass
            self._on_editor_workflow_changed()
        return bool(ok)

    def _on_stop_workflow(self):
        self._toolbar.set_run_controls_state(
            run_enabled=False,
            stop_enabled=False,
            stop_text=tr_("⏳ Stopping..."),
        )

        event_bus = get_event_bus()
        event_bus.emit_status(tr_("Stopping workflow... (waiting for current task to finish)"))
        event_bus.training_stopped.emit()

        if self._controller is not None:
            self._controller.stop()

    def _fit_current(self):
        editor = self._current_editor()
        if editor:
            try:
                editor.fit_to_selection()
            except Exception:
                pass

    def _clear_current(self):
        editor = self._current_editor()
        if not editor:
            return

        reply = QMessageBox.question(
            self,
            tr_("Confirm"),
            tr_("Clear the current workflow?"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        workflow = Workflow("new_workflow")
        workflow._file_path = None  # type: ignore[attr-defined]
        editor.set_workflow(workflow)
        self._on_editor_workflow_changed()

    def _on_tab_close_requested(self, index: int):
        tab = self._get_tab(index)
        if not tab:
            return

        editor = tab.editor
        workflow = editor.get_workflow()
        if workflow and workflow.is_dirty:
            name = self._workflow_display_name(workflow)
            msg = tr_("Save changes to '{name}' before closing?").format(name=name)
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Question)
            box.setWindowTitle(tr_("Confirm"))
            box.setText(msg)
            box.setStandardButtons(
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel
            )
            box.setDefaultButton(QMessageBox.StandardButton.Save)
            choice = box.exec()

            if choice == QMessageBox.StandardButton.Cancel:
                return
            if choice == QMessageBox.StandardButton.Save:
                if not self._save_editor(editor):
                    return

        self._tabs.removeTab(index)
        editor.deleteLater()

        if self._tabs.count() == 0:
            self.new_tab()
        else:
            self._refresh_titles()
            self.workflow_changed.emit()
