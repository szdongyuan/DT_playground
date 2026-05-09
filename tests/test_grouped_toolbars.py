import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QWidget

from src.ui.views.model_builder_view import ModelBuilderView
from src.ui.views.workflow_toolbar import WorkflowToolbar


def _texts_for_labels(parent: QWidget, prefix: str) -> list[str]:
    return [
        label.text()
        for label in parent.findChildren(QLabel)
        if label.objectName().startswith(prefix)
    ]


def _button(parent: QWidget, object_name: str) -> QPushButton:
    button = parent.findChild(QPushButton, object_name)
    assert button is not None, object_name
    return button


def test_workflow_toolbar_uses_grouped_command_layout():
    app = QApplication.instance() or QApplication([])
    toolbar = WorkflowToolbar(show_duplicate=True)

    try:
        assert toolbar.objectName() == "workflowCommandToolbar"
        assert 54 <= toolbar.height() <= 64
        assert _texts_for_labels(toolbar, "toolbarGroupTitle_") == [
            "文件",
            "编辑",
            "运行",
            "视图",
        ]

        expected_buttons = {
            "workflowToolbarButton_new": "新建",
            "workflowToolbarButton_open": "打开",
            "workflowToolbarButton_save": "保存",
            "workflowToolbarButton_save_as": "另存为",
            "workflowToolbarButton_copy": "复制",
            "workflowToolbarButton_delete": "删除",
            "workflowToolbarButton_run": "运行",
            "workflowToolbarButton_stop": "停止",
            "workflowToolbarButton_fit": "适应",
        }
        for object_name, label in expected_buttons.items():
            assert label in _button(toolbar, object_name).text()

        assert _button(toolbar, "workflowToolbarButton_save_as").isEnabled()
        assert _button(toolbar, "workflowToolbarButton_copy").isEnabled()
        assert _button(toolbar, "workflowToolbarButton_delete").isEnabled()
    finally:
        toolbar.close()
        app.processEvents()


def test_model_builder_toolbar_uses_matching_grouped_command_layout():
    app = QApplication.instance() or QApplication([])
    view = ModelBuilderView()

    try:
        toolbar = view.findChild(QWidget, "modelCommandToolbar")
        assert toolbar is not None
        assert 54 <= toolbar.height() <= 64
        assert _texts_for_labels(toolbar, "toolbarGroupTitle_") == [
            "文件",
            "编辑",
            "模型",
            "视图",
        ]

        expected_buttons = {
            "modelToolbarButton_new": "新建",
            "modelToolbarButton_open": "打开",
            "modelToolbarButton_save": "保存",
            "modelToolbarButton_save_as": "另存为",
            "modelToolbarButton_copy": "复制",
            "modelToolbarButton_delete": "删除",
            "modelToolbarButton_build": "构建模型",
            "modelToolbarButton_import_keras": "导入Keras",
            "modelToolbarButton_fit": "适应",
        }
        for object_name, label in expected_buttons.items():
            assert label in _button(toolbar, object_name).text()
    finally:
        view.close()
        app.processEvents()
