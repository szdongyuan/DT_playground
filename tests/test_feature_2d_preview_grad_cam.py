import os

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from src.ui.widgets.preview.feature_2d_preview import Feature2DPreviewWidget
from src.workflow.nodes.feature import FeatureData


def test_feature_2d_preview_marks_grad_cam_heatmap():
    app = QApplication.instance() or QApplication([])
    widget = Feature2DPreviewWidget()

    try:
        feature = FeatureData(
            data=np.array([[[0.0, 0.5, 1.0], [0.2, 0.4, 0.8]]], dtype=np.float32),
            feature_type="grad_cam",
            sample_rate=48000,
            hop_length=512,
        )

        assert widget.set_data(feature)

        assert "type=grad_cam" in widget.data_info
        assert widget.data_info.startswith("🔥 FEATURE_2D")
        assert widget._heatmap_group.title() == "Grad-CAM heatmap"
        assert "stronger class evidence" in widget._stats_label.text()
    finally:
        widget.close()
        app.processEvents()


def test_feature_2d_preview_marks_grad_cam_overlay():
    app = QApplication.instance() or QApplication([])
    widget = Feature2DPreviewWidget()

    try:
        feature = FeatureData(
            data=np.array([[[0.0, 0.5], [0.75, 1.0]]], dtype=np.float32),
            feature_type="grad_cam_overlay",
            sample_rate=48000,
            hop_length=512,
        )

        assert widget.set_data(feature)

        assert widget.data_info.startswith("🧩 FEATURE_2D")
        assert widget._heatmap_group.title() == "Grad-CAM overlay"
        assert "blended with Grad-CAM intensity" in widget._stats_label.text()
    finally:
        widget.close()
        app.processEvents()
