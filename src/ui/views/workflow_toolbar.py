# -*- coding: utf-8 -*-
"""
Workflow Toolbar

Reusable toolbar component for workflow views.
Provides title label, command buttons and breakpoint mode toggle.
Communicates purely via signals -- contains no workflow / editor logic.
"""

from typing import Optional

from PySide6.QtCore import Signal
from ..i18n import tr_
from ..widgets.command_toolbar import GroupedCommandToolbar, command_button


class WorkflowToolbar(GroupedCommandToolbar):
    """
    Reusable workflow toolbar.

    Signals are emitted when the user clicks the corresponding button.
    The container / parent is responsible for connecting them to actual logic.
    """

    new_requested = Signal()
    open_requested = Signal()
    save_requested = Signal()
    save_as_requested = Signal()
    duplicate_requested = Signal()
    copy_requested = Signal()
    delete_requested = Signal()
    run_requested = Signal()
    stop_requested = Signal()
    continue_requested = Signal()
    fit_requested = Signal()
    clear_requested = Signal()

    def __init__(self, parent=None, *, show_duplicate: bool = False):
        super().__init__("workflowCommandToolbar", parent)
        self._show_duplicate = show_duplicate
        self._breakpoint_active = False
        self._setup_ui()

    # ===== Public API =====

    def set_title(self, text: str):
        self.setToolTip(text)

    def set_run_controls_state(
        self,
        *,
        run_enabled: bool,
        stop_enabled: bool,
        stop_text: Optional[str] = None,
    ) -> None:
        self._run_btn.setEnabled(run_enabled)
        self._stop_btn.setEnabled(stop_enabled)
        if stop_text is not None:
            self._stop_btn.setText(stop_text)
        elif not self._stop_btn.text().endswith(tr_("停止")):
            self._stop_btn.setText(tr_("⏹ 停止"))

    def show_breakpoint_mode(self, enabled: bool, node_name: str = ""):
        self._breakpoint_active = enabled
        if enabled:
            suffix = f" ({node_name})" if node_name else ""
            self._run_btn.setText(tr_("⏵ 继续") + suffix)
            self._run_btn.setEnabled(True)
            return
        self._run_btn.setText(tr_("▶ 运行"))

    # ===== UI construction =====

    def _setup_ui(self):
        new_btn = command_button(tr_("📄 新建"), "workflowToolbarButton_new", parent=self)
        new_btn.clicked.connect(self.new_requested)
        open_btn = command_button(tr_("📂 打开"), "workflowToolbarButton_open", parent=self)
        open_btn.clicked.connect(self.open_requested)
        save_btn = command_button(tr_("💾 保存"), "workflowToolbarButton_save", parent=self)
        save_btn.clicked.connect(self.save_requested)
        save_as_btn = command_button(tr_("📋 另存为"), "workflowToolbarButton_save_as", parent=self)
        save_as_btn.clicked.connect(self.save_as_requested)
        self.add_group("file", tr_("文件"), [new_btn, open_btn, save_btn, save_as_btn])

        copy_btn = command_button(tr_("⧉ 复制"), "workflowToolbarButton_copy", parent=self)
        copy_btn.clicked.connect(self.copy_requested)
        delete_btn = command_button(
            tr_("🗑 删除"),
            "workflowToolbarButton_delete",
            variant="danger",
            parent=self,
        )
        delete_btn.clicked.connect(self.delete_requested)
        self.add_group("edit", tr_("编辑"), [copy_btn, delete_btn])

        self._run_btn = command_button(
            tr_("▶ 运行"),
            "workflowToolbarButton_run",
            variant="primary",
            parent=self,
        )
        self._run_btn.clicked.connect(self._emit_primary_run_action)
        self._stop_btn = command_button(
            tr_("⏹ 停止"),
            "workflowToolbarButton_stop",
            variant="danger",
            enabled=False,
            parent=self,
        )
        self._stop_btn.clicked.connect(self.stop_requested)
        self.add_group("run", tr_("运行"), [self._run_btn, self._stop_btn])

        fit_btn = command_button(tr_("⛶ 适应"), "workflowToolbarButton_fit", parent=self)
        fit_btn.clicked.connect(self.fit_requested)
        self.add_group("view", tr_("视图"), [fit_btn])
        self.add_end_stretch()

    def _emit_primary_run_action(self) -> None:
        if self._breakpoint_active:
            self.continue_requested.emit()
        else:
            self.run_requested.emit()
