import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from src.ui.views.preview_view import PreviewView
from src.ui.widgets.preview.anomaly_result_preview import AnomalyResultPreviewWidget
from src.workflow.nodes.anomaly import AnomalyResultData, AnomalyScoresData
from src.workflow.nodes.feature import FeatureData


def _result():
    feature = FeatureData(
        data=np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32),
        feature_type="mel_spectrogram",
        sample_rate=16000,
        hop_length=160,
        source_file="machine.wav",
    )
    scores = AnomalyScoresData(
        raw_scores=np.array([0.2, 0.8]),
        normalized_scores=np.array([20.0, 90.0]),
        sample_ids=["normal.wav", "machine.wav"],
        source_items=[feature, feature],
        reference_normalized_scores=np.array([5.0, 10.0, 20.0]),
    )
    return AnomalyResultData(
        scores=scores,
        threshold=80.0,
        attention_threshold=60.0,
        severities=["normal", "anomaly"],
        is_anomaly=np.array([False, True]),
        strategy="manual",
    )


def test_anomaly_result_preview_shows_summary_ranking_and_selected_feature():
    app = QApplication.instance() or QApplication([])
    widget = AnomalyResultPreviewWidget()

    try:
        assert widget.set_data(_result())
        assert "samples=2" in widget.data_info
        assert "Anomalies: 1" in widget._summary_label.text()
        assert widget._table.rowCount() == 2
        assert widget._table.item(0, 1).text() == "machine.wav"
        assert widget._detail_stack.currentWidget() is widget._feature_2d_preview
    finally:
        widget.close()
        app.processEvents()


def test_preview_view_routes_anomaly_results_to_explorer():
    app = QApplication.instance() or QApplication([])
    view = PreviewView()

    try:
        view.set_node_data("decision_1", "Anomaly decision", {"anomaly_result": _result()})

        assert view._stack.currentWidget() is view._preview_widgets["anomaly_result"]
        assert "ANOMALY_RESULT" in view._dim_info.text()
    finally:
        view.close()
        app.processEvents()
