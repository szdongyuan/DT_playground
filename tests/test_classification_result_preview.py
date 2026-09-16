import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from src.ui.views.preview_view import PreviewView
from src.ui.widgets.preview.classification_result_preview import (
    ClassificationResultPreviewWidget,
)


def _result():
    return {
        "schema_version": "1.0",
        "task": "classification",
        "sample_count": 2,
        "classes": [{"id": 0, "name": "cat"}, {"id": 1, "name": "dog"}],
        "metrics": {
            "loss": 0.25,
            "accuracy": 0.5,
            "balanced_accuracy": 0.5,
            "macro_precision": 0.25,
            "macro_recall": 0.5,
            "macro_f1": 1 / 3,
            "weighted_precision": 0.25,
            "weighted_recall": 0.5,
            "weighted_f1": 1 / 3,
        },
        "confusion_matrix": [[1, 0], [1, 0]],
        "per_class": [
            {
                "class_id": 0,
                "class_name": "cat",
                "precision": 0.5,
                "recall": 1.0,
                "f1": 2 / 3,
                "support": 1,
            },
            {
                "class_id": 1,
                "class_name": "dog",
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "support": 1,
            },
        ],
        "predictions": [
            {
                "index": 0,
                "file_path": "cat.wav",
                "true_class_id": 0,
                "true_class_name": "cat",
                "predicted_class_id": 0,
                "predicted_class_name": "cat",
                "confidence": 0.8,
            },
            {
                "index": 1,
                "file_path": "dog.wav",
                "true_class_id": 1,
                "true_class_name": "dog",
                "predicted_class_id": 0,
                "predicted_class_name": "cat",
                "confidence": 0.6,
            },
        ],
        "warnings": [{"code": "unpredicted_classes", "class_ids": [1]}],
    }


def test_classification_result_preview_shows_all_research_views():
    app = QApplication.instance() or QApplication([])
    widget = ClassificationResultPreviewWidget()
    try:
        assert widget.set_data(_result())
        assert "samples=2 classes=2" in widget.data_info
        assert "accuracy" in widget._summary_label.text()
        assert "dog" in widget._warnings_label.text()
        assert widget._per_class_table.rowCount() == 2
        assert widget._confusion_table.rowCount() == 2
        assert widget._confusion_table.item(1, 0).text() == "1"
        assert widget._predictions_table.rowCount() == 2
        assert widget._predictions_table.item(1, 1).text() == "dog.wav"
        widget._only_errors.setChecked(True)
        assert widget._predictions_table.rowCount() == 1
        assert widget._predictions_table.item(0, 1).text() == "dog.wav"
    finally:
        widget.close()
        app.processEvents()


def test_preview_view_routes_classification_result_to_specialized_preview():
    app = QApplication.instance() or QApplication([])
    view = PreviewView()
    try:
        view.set_node_data(
            "evaluation_1",
            "Classification model evaluation",
            {"classification_result": _result()},
        )
        assert (
            view._stack.currentWidget()
            is view._preview_widgets["classification_result"]
        )
        assert "CLASSIFICATION_RESULT" in view._dim_info.text()
    finally:
        view.close()
        app.processEvents()
