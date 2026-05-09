# -*- coding: utf-8 -*-
"""Model builder page toolbar declaration."""

from PySide6.QtCore import Signal

from src.ui.i18n import tr_
from src.ui.widgets.command_toolbar import GroupedCommandToolbar
from src.ui.widgets.command_toolbar_spec import ToolbarActionSpec, ToolbarGroupSpec


MODEL_TOOLBAR_GROUPS = (
    ToolbarGroupSpec(
        key="file",
        title=tr_("文件"),
        actions=(
            ToolbarActionSpec("new", tr_("📄 新建"), "modelToolbarButton_new"),
            ToolbarActionSpec("open", tr_("📂 打开"), "modelToolbarButton_open"),
            ToolbarActionSpec("save", tr_("💾 保存"), "modelToolbarButton_save"),
            ToolbarActionSpec("save_as", tr_("📋 另存为"), "modelToolbarButton_save_as"),
        ),
    ),
    ToolbarGroupSpec(
        key="edit",
        title=tr_("编辑"),
        actions=(
            ToolbarActionSpec("copy", tr_("⧉ 复制"), "modelToolbarButton_copy"),
            ToolbarActionSpec(
                "delete",
                tr_("🗑 删除"),
                "modelToolbarButton_delete",
                variant="danger",
            ),
        ),
    ),
    ToolbarGroupSpec(
        key="model",
        title=tr_("模型"),
        actions=(
            ToolbarActionSpec(
                "build",
                tr_("🔨 构建模型"),
                "modelToolbarButton_build",
                variant="primary",
            ),
            ToolbarActionSpec(
                "import_keras",
                tr_("📥 导入Keras"),
                "modelToolbarButton_import_keras",
                variant="warning",
                tooltip=tr_("Import architecture from a trained .keras/.h5 model file"),
            ),
        ),
    ),
    ToolbarGroupSpec(
        key="view",
        title=tr_("视图"),
        actions=(ToolbarActionSpec("fit", tr_("⛶ 适应"), "modelToolbarButton_fit"),),
    ),
)


class ModelToolbar(GroupedCommandToolbar):
    """Reusable model toolbar that exposes page-level command signals."""

    new_requested = Signal()
    open_requested = Signal()
    save_requested = Signal()
    save_as_requested = Signal()
    copy_requested = Signal()
    delete_requested = Signal()
    build_requested = Signal()
    import_keras_requested = Signal()
    fit_requested = Signal()

    def __init__(self, parent=None):
        super().__init__("modelCommandToolbar", parent)
        self._buttons = self.build_from_spec(MODEL_TOOLBAR_GROUPS)
        self._connect_buttons()

    def _connect_buttons(self) -> None:
        self._buttons["new"].clicked.connect(self.new_requested)
        self._buttons["open"].clicked.connect(self.open_requested)
        self._buttons["save"].clicked.connect(self.save_requested)
        self._buttons["save_as"].clicked.connect(self.save_as_requested)
        self._buttons["copy"].clicked.connect(self.copy_requested)
        self._buttons["delete"].clicked.connect(self.delete_requested)
        self._buttons["build"].clicked.connect(self.build_requested)
        self._buttons["import_keras"].clicked.connect(self.import_keras_requested)
        self._buttons["fit"].clicked.connect(self.fit_requested)
