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


@dataclass
class _RunControlsState:
    run_enabled: bool
    stop_enabled: bool
    stop_text: Optional[str] = None


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
    run_from_node_requested = pyqtSignal(str)
    continue_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._controller: Optional[WorkflowController] = None
        self._running_tab_index: Optional[int] = None
        # Guard session persistence during startup/session restore. The tab view is
        # constructed before app settings are loaded, and it always creates a
        # default empty tab. Without this guard the empty tab would overwrite the
        # previous session's open_workflow_paths on disk.
        self._session_restore_in_progress: bool = True
        self._running_controls: _RunControlsState = _RunControlsState(
            run_enabled=True,
            stop_enabled=False,
            stop_text=None,
        )
        self._breakpoint_active: bool = False
        self._breakpoint_node_name: str = ""

        self._setup_ui()
        self._connect_signals()

        # Always keep at least one tab.
        self.new_tab()

    # ===== Session persistence / app-close helpers =====

    def persist_session_state(self) -> None:
        """
        Persist workflow tab session state.

        Stores only file-backed workflow tabs (existing paths) to:
        - session.open_workflow_paths: list[str]
        - session.active_workflow_tab: int (index into open_workflow_paths, or -1)
        """
        self.persist_session_state_guarded()

    def set_session_restore_in_progress(self, enabled: bool) -> None:
        self._session_restore_in_progress = bool(enabled)

    def persist_session_state_guarded(self, *, force: bool = False) -> None:
        """
        Persist workflow tab session state with a startup/restore guard.

        During app startup/session restore, the view may temporarily contain only
        a default empty tab. Persisting at that time would wipe the previous
        session's open_workflow_paths. Call with force=True only when the app is
        closing and we want to persist the final state.
        """
        if self._session_restore_in_progress and not force:
            return

        paths, active = self._compute_file_backed_session_state()
        try:
            config.set("session.open_workflow_paths", paths)
            config.set("session.active_workflow_tab", active)
        except Exception:
            pass

    def confirm_save_dirty_on_close(self) -> bool:
        """
        Prompt the user if there are dirty workflows before closing the app.

        Returns True if the app can close, False if the close should be cancelled.
        """
        dirty_editors = self._get_dirty_editors()
        if not dirty_editors:
            return True

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle(tr_("Confirm"))
        box.setText(tr_("There are unsaved workflows. Save your changes before closing?"))
        box.setInformativeText(tr_("If you don't save, your changes will be lost."))
        box.setStandardButtons(
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel
        )
        box.setDefaultButton(QMessageBox.StandardButton.Save)

        choice = box.exec()
        if choice == QMessageBox.StandardButton.Cancel:
            return False
        if choice == QMessageBox.StandardButton.Discard:
            return True

        # Save: save all dirty workflows; any cancellation/failure cancels close.
        for editor in dirty_editors:
            if not self._save_editor(editor):
                return False

        self._refresh_all_tab_titles()
        return True

    def clear_all_tabs(self, *, ensure_one_tab: bool = True) -> None:
        """Remove all tabs without prompting (used for session restore)."""
        while self._tabs.count() > 0:
            w = self._tabs.widget(0)
            self._tabs.removeTab(0)
            if w is not None:
                try:
                    w.deleteLater()
                except Exception:
                    pass

        if ensure_one_tab:
            self.new_tab()

        self.persist_session_state_guarded()

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
        self._set_editor_session_file_path(editor, getattr(workflow, "_file_path", None))
        self._refresh_titles()
        self.workflow_changed.emit()

    def get_workflow(self) -> Optional[Workflow]:
        editor = self._current_editor()
        return editor.get_workflow() if editor else None

    def reset_all_node_states(self):
        editor = self._runtime_editor()
        if editor:
            editor.reset_all_node_states()

    def reset_node_states(self, node_ids: list[str]):
        editor = self._runtime_editor()
        if editor:
            editor.reset_node_states(node_ids)

    def update_node_state(self, node_id: str, state: str):
        editor = self._runtime_editor()
        if editor:
            editor.update_node_state(node_id, state)

    def highlight_node(self, node_id: str):
        editor = self._runtime_editor()
        if editor:
            editor.highlight_node(node_id)

    def show_breakpoint_mode(self, enabled: bool, node_name: str = ""):
        self._breakpoint_active = bool(enabled)
        self._breakpoint_node_name = node_name or ""
        self._apply_toolbar_state_for_current_tab()

    def set_run_controls_state(
        self,
        *,
        run_enabled: bool,
        stop_enabled: bool,
        stop_text: Optional[str] = None,
    ) -> None:
        if self._running_tab_index is None:
            self._toolbar.set_run_controls_state(
                run_enabled=run_enabled,
                stop_enabled=stop_enabled,
                stop_text=stop_text,
            )
            return

        self._running_controls = _RunControlsState(
            run_enabled=run_enabled,
            stop_enabled=stop_enabled,
            stop_text=stop_text,
        )
        self._apply_toolbar_state_for_current_tab()

    # ===== Running tab tracking =====

    def get_current_tab_index(self) -> int:
        return int(self._tabs.currentIndex())

    def get_running_tab_index(self) -> Optional[int]:
        return self._running_tab_index

    def set_running_tab_index(self, index: Optional[int]) -> None:
        if index is None:
            self._running_tab_index = None
            self._apply_toolbar_state_for_current_tab()
            return

        if 0 <= index < self._tabs.count():
            self._running_tab_index = int(index)
        else:
            self._running_tab_index = None
        self._apply_toolbar_state_for_current_tab()

    def clear_running_tab(self) -> None:
        self.set_running_tab_index(None)

    def clear_breakpoint_mode(self) -> None:
        self._breakpoint_active = False
        self._breakpoint_node_name = ""
        self._apply_toolbar_state_for_current_tab()

    def activate_running_tab(self) -> None:
        idx = self._running_tab_index
        if idx is None:
            return
        if 0 <= idx < self._tabs.count():
            self._tabs.setCurrentIndex(idx)

    def is_current_tab_running(self) -> bool:
        idx = self._running_tab_index
        return idx is not None and idx == self._tabs.currentIndex()

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
        # If the current tab is a pristine empty workflow, reuse it instead of creating
        # an extra tab. This avoids keeping an unused "new_workflow" tab around.
        editor = self._current_editor()
        current = editor.get_workflow() if editor else None
        can_reuse_current = False
        try:
            can_reuse_current = bool(
                editor is not None
                and current is not None
                and (not getattr(current, "is_dirty", False))
                and (not getattr(current, "_file_path", None))
                and (len(getattr(current, "nodes", {})) == 0)
                and (len(getattr(current, "connections", [])) == 0)
            )
        except Exception:
            can_reuse_current = False

        if can_reuse_current and editor is not None:
            editor.set_workflow(workflow)
            self._set_editor_session_file_path(editor, path)
            self._refresh_all_tab_titles()
            self.workflow_changed.emit()
            self.persist_session_state_guarded()
        else:
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
        self._set_editor_session_file_path(editor, getattr(workflow, "_file_path", None))

        editor.workflow_changed.connect(self._on_editor_workflow_changed)
        editor.node_selected.connect(self.node_selected)
        editor.node_double_clicked.connect(self.node_double_clicked)
        editor.run_from_node_requested.connect(self.run_from_node_requested)

        idx = self._tabs.addTab(editor, "")
        self._update_tab_title(idx)
        if make_current:
            self._tabs.setCurrentIndex(idx)
            self._refresh_titles()
            self.workflow_changed.emit()
            self.persist_session_state_guarded()

    def _get_tab(self, index: int) -> Optional[_Tab]:
        w = self._tabs.widget(index)
        if isinstance(w, WorkflowEditorWidget):
            return _Tab(editor=w)
        return None

    def _current_editor(self) -> Optional[WorkflowEditorWidget]:
        tab = self._get_tab(self._tabs.currentIndex())
        return tab.editor if tab else None

    def _runtime_editor(self) -> Optional[WorkflowEditorWidget]:
        idx = self._running_tab_index
        if idx is not None:
            tab = self._get_tab(idx)
            if tab:
                return tab.editor
        return self._current_editor()

    def _apply_toolbar_state_for_current_tab(self) -> None:
        """
        Keep toolbar controls consistent with the selected tab.

        While a workflow is running (or stopping/paused), only the running tab can
        interact with runtime controls. Other tabs show disabled controls.
        """
        if self._running_tab_index is None:
            if self._breakpoint_active:
                self._toolbar.show_breakpoint_mode(False)
            return

        if self.is_current_tab_running():
            self._toolbar.show_breakpoint_mode(
                self._breakpoint_active,
                self._breakpoint_node_name,
            )
            if not self._breakpoint_active:
                self._toolbar.set_run_controls_state(
                    run_enabled=self._running_controls.run_enabled,
                    stop_enabled=self._running_controls.stop_enabled,
                    stop_text=self._running_controls.stop_text,
                )
            return

        self._toolbar.show_breakpoint_mode(False)
        self._toolbar.set_run_controls_state(
            run_enabled=False,
            stop_enabled=False,
            stop_text=None,
        )

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

    def _refresh_all_tab_titles(self) -> None:
        for i in range(self._tabs.count()):
            self._update_tab_title(i)
        self._refresh_titles()

    # ===== Event handlers =====

    def _on_editor_workflow_changed(self):
        self._refresh_titles()
        self.workflow_changed.emit()

    def _on_current_changed(self, index: int):
        self._update_tab_title(index)
        self._refresh_titles()
        self._apply_toolbar_state_for_current_tab()
        self.workflow_changed.emit()
        self.persist_session_state_guarded()

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
                self._set_editor_session_file_path(editor, file_path)
                try:
                    config.set("session.last_workflow_path", file_path)
                    config.add_recent_file(file_path)
                except Exception:
                    pass
                self._refresh_all_tab_titles()
                self.persist_session_state_guarded()
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
            self._set_editor_session_file_path(editor, path)
            try:
                config.set("session.last_workflow_path", path)
                config.add_recent_file(path)
            except Exception:
                pass
            self._refresh_all_tab_titles()
            self.persist_session_state_guarded()
        return bool(ok)

    def _on_stop_workflow(self):
        self.set_run_controls_state(
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
        if self._running_tab_index is not None:
            if index == self._running_tab_index:
                QMessageBox.information(
                    self,
                    tr_("Running"),
                    tr_("Cannot close the running tab while a workflow is active."),
                )
                return
            if index < self._running_tab_index:
                self._running_tab_index -= 1

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
            self._apply_toolbar_state_for_current_tab()
            self.workflow_changed.emit()
        self.persist_session_state_guarded()

    # ===== Session state internals =====

    def _normalize_existing_path(self, path: str) -> Optional[str]:
        if not isinstance(path, str):
            return None
        p = path.strip()
        if not p:
            return None
        p = os.path.abspath(p)
        if os.path.exists(p):
            return p
        return None

    def _set_editor_session_file_path(self, editor: WorkflowEditorWidget, file_path: object) -> None:
        """
        Persist a per-tab file path independent of Workflow internals.

        This is more robust than relying solely on `workflow._file_path` because some
        callsites may replace workflow objects or forget to attach the path.
        """
        p: Optional[str] = None
        if isinstance(file_path, str):
            p = self._normalize_existing_path(file_path)
        try:
            editor.setProperty("workflow_file_path", p or "")
        except Exception:
            pass

    def _get_editor_session_file_path(self, editor: WorkflowEditorWidget) -> Optional[str]:
        try:
            raw = editor.property("workflow_file_path")
        except Exception:
            raw = None
        if isinstance(raw, str) and raw.strip():
            return self._normalize_existing_path(raw)
        return None

    def _compute_file_backed_session_state(self) -> tuple[list[str], int]:
        paths: list[str] = []
        active = -1
        current_tab = int(self._tabs.currentIndex())

        for tab_idx in range(self._tabs.count()):
            tab = self._get_tab(tab_idx)
            if not tab:
                continue
            file_path = self._get_editor_session_file_path(tab.editor)
            if not file_path:
                wf = tab.editor.get_workflow()
                if not wf:
                    continue
                raw = getattr(wf, "_file_path", None)
                file_path = self._normalize_existing_path(raw) if isinstance(raw, str) else None
            if not file_path:
                continue

            if tab_idx == current_tab:
                active = len(paths)
            paths.append(file_path)

        return paths, active

    def _get_dirty_editors(self) -> list[WorkflowEditorWidget]:
        dirty: list[WorkflowEditorWidget] = []
        for i in range(self._tabs.count()):
            tab = self._get_tab(i)
            if not tab:
                continue
            wf = tab.editor.get_workflow()
            if wf and getattr(wf, "is_dirty", False):
                dirty.append(tab.editor)
        return dirty
