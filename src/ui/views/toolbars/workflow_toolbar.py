# -*- coding: utf-8 -*-
"""Workflow page toolbar declaration."""

from typing import Optional

from PySide6.QtCore import Signal

from src.ui.i18n import tr_
from src.ui.widgets.command_toolbar import GroupedCommandToolbar
from src.ui.widgets.command_toolbar_spec import ToolbarActionSpec, ToolbarGroupSpec


WORKFLOW_TOOLBAR_GROUPS = (
    ToolbarGroupSpec(
        key="file",
        title=tr_("文件"),
        actions=(
            ToolbarActionSpec("new", tr_("📄 新建"), "workflowToolbarButton_new"),
            ToolbarActionSpec("open", tr_("📂 打开"), "workflowToolbarButton_open"),
            ToolbarActionSpec("save", tr_("💾 保存"), "workflowToolbarButton_save"),
            ToolbarActionSpec("save_as", tr_("📋 另存为"), "workflowToolbarButton_save_as"),
        ),
    ),
    ToolbarGroupSpec(
        key="edit",
        title=tr_("编辑"),
        actions=(
            ToolbarActionSpec("copy", tr_("⧉ 复制"), "workflowToolbarButton_copy"),
            ToolbarActionSpec(
                "delete",
                tr_("🗑 删除"),
                "workflowToolbarButton_delete",
                variant="danger",
            ),
        ),
    ),
    ToolbarGroupSpec(
        key="run",
        title=tr_("运行"),
        actions=(
            ToolbarActionSpec(
                "run",
                tr_("▶ 运行"),
                "workflowToolbarButton_run",
                variant="primary",
            ),
            ToolbarActionSpec(
                "stop",
                tr_("⏹ 停止"),
                "workflowToolbarButton_stop",
                variant="danger",
                enabled=False,
            ),
        ),
    ),
    ToolbarGroupSpec(
        key="view",
        title=tr_("视图"),
        actions=(ToolbarActionSpec("fit", tr_("⛶ 适应"), "workflowToolbarButton_fit"),),
    ),
)


class WorkflowToolbar(GroupedCommandToolbar):
    """Reusable workflow toolbar that exposes page-level command signals."""

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
        self._buttons = self.build_from_spec(WORKFLOW_TOOLBAR_GROUPS)
        self._run_btn = self._buttons["run"]
        self._stop_btn = self._buttons["stop"]
        self._connect_buttons()

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

    def _connect_buttons(self) -> None:
        self._buttons["new"].clicked.connect(self.new_requested)
        self._buttons["open"].clicked.connect(self.open_requested)
        self._buttons["save"].clicked.connect(self.save_requested)
        self._buttons["save_as"].clicked.connect(self.save_as_requested)
        self._buttons["copy"].clicked.connect(self.copy_requested)
        self._buttons["delete"].clicked.connect(self.delete_requested)
        self._buttons["run"].clicked.connect(self._emit_primary_run_action)
        self._buttons["stop"].clicked.connect(self.stop_requested)
        self._buttons["fit"].clicked.connect(self.fit_requested)

    def _emit_primary_run_action(self) -> None:
        if self._breakpoint_active:
            self.continue_requested.emit()
        else:
            self.run_requested.emit()
