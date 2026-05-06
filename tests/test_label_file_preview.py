import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from src.controllers.navigation_controller import NavigationController
from src.controllers.navigation_dto import NodeDoubleClickContext
from src.controllers.view_types import ViewType
from src.ui.widgets.preview.label_preview import LabelPreviewWidget


def test_label_file_double_click_opens_preview_when_outputs_exist():
    app = QApplication.instance() or QApplication([])
    controller = NavigationController()

    outputs = {
        "labels": [0, 1, 0],
        "label_map": {
            "filename_map": {
                "cat_001.wav": 0,
                "dog_001.wav": 1,
                "cat_002.wav": 0,
            },
            "label_names": {
                "cat": 0,
                "dog": 1,
            },
        },
    }
    ctx = NodeDoubleClickContext(
        node_id="label_file_1",
        node_type="label_file",
        category="data_source",
        display_name="Label file",
        has_output=True,
        outputs=outputs,
    )

    decision = controller.decide_node_double_click(ctx)

    assert decision.handled is True
    assert decision.message_box is None
    assert decision.show_not_run_tip is False
    assert decision.switch_to == ViewType.PREVIEW
    assert decision.preview_payload == ("label_file_1", "Label file", outputs)
    app.processEvents()


def test_label_preview_displays_filename_label_and_class_name():
    app = QApplication.instance() or QApplication([])
    widget = LabelPreviewWidget()

    try:
        assert widget.set_data(
            {
                "filename_map": {
                    "cat_001.wav": 0,
                    "dog_001.wav": 1,
                    "cat_002.wav": 0,
                },
                "label_names": {
                    "cat": 0,
                    "dog": 1,
                },
            }
        )

        assert widget._table.columnCount() == 3
        assert widget._table.item(0, 0).text() == "cat_001.wav"
        assert widget._table.item(0, 1).text() == "0"
        assert widget._table.item(0, 2).text() == "cat"
        assert widget._table.item(1, 0).text() == "dog_001.wav"
        assert widget._table.item(1, 1).text() == "1"
        assert widget._table.item(1, 2).text() == "dog"
    finally:
        widget.close()
        app.processEvents()


def test_label_preview_keeps_plain_filename_label_map_items():
    app = QApplication.instance() or QApplication([])
    widget = LabelPreviewWidget()

    try:
        assert widget.set_data(
            {
                "cat_001.wav": 0,
                "dog_001.wav": 1,
            }
        )

        assert widget._table.item(0, 0).text() == "cat_001.wav"
        assert widget._table.item(0, 1).text() == "0"
        assert widget._table.item(1, 0).text() == "dog_001.wav"
        assert widget._table.item(1, 1).text() == "1"
    finally:
        widget.close()
        app.processEvents()
