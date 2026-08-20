import csv
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QDialog

import src.ui.widgets.preview.multi_curve_preview as multi_curve_preview_module
from src.ui.node_editor.property_panel import PropertyPanel
from src.ui.styles import Styles
from src.ui.views.preview_view import PreviewView
from src.ui.widgets.preview.multi_curve_export import (
    CurveExportItem,
    MultiCurveExportDialog,
    write_curve_csv,
)
from src.ui.widgets.preview.multi_curve_preview import MultiCurvePreviewWidget
from src.workflow.node_base import NodeCategory, create_node
from src.workflow.nodes.data_source import AudioData
from src.workflow.nodes.feature import FeatureData
from src.workflow.nodes.visualization import (
    MultiCurvePreviewData,
    prepare_multi_curve_preview,
)
from src.workflow.port import DataType


def _settings(**overrides):
    settings = {
        "interaction_mode": "performance",
        "display_mode": "overlay",
        "x_alignment": "normalized",
        "channel_mode": "separate",
        "selected_channel": 0,
        "value_transform": "none",
        "aggregate_mode": "mean",
        "variability_mode": "std",
        "percentile_low": 25.0,
        "percentile_high": 75.0,
        "max_points": 2000,
        "line_opacity": 0.3,
        "show_legend": True,
    }
    settings.update(overrides)
    return settings


def test_multi_curve_viewer_is_terminal_output_node():
    node = create_node("multi_curve_viewer")

    assert node is not None
    assert node.category == NodeCategory.OUTPUT
    assert list(node.inputs) == ["data"]
    assert node.inputs["data"].data_type == DataType.ANY
    assert node.outputs == {}
    assert node.get_parameter("interaction_mode") == "performance"
    assert node.get_parameter("show_legend") is False


def test_normalized_alignment_interpolates_lengths_before_aggregate():
    preview = prepare_multi_curve_preview(
        [np.array([0.0, 1.0, 2.0]), np.array([2.0, 2.0])],
        _settings(),
    )

    assert len(preview.series) == 2
    assert preview.series[0].x is preview.series[1].x
    assert preview.series[0].x.dtype == np.float32
    assert all(series.y.dtype == np.float32 for series in preview.series)
    assert preview.aggregate_y.dtype == np.float32
    assert preview.lower_y.dtype == np.float32
    assert preview.upper_y.dtype == np.float32
    np.testing.assert_allclose(preview.series[0].x, [0.0, 50.0, 100.0])
    np.testing.assert_allclose(preview.series[1].y, [2.0, 2.0, 2.0])
    np.testing.assert_allclose(preview.aggregate_y, [1.0, 1.5, 2.0])
    np.testing.assert_allclose(preview.lower_y, preview.aggregate_y - [1.0, 0.5, 0.0])
    np.testing.assert_allclose(preview.upper_y, preview.aggregate_y + [1.0, 0.5, 0.0])


def test_feature_map_is_rejected_instead_of_flattened():
    feature = FeatureData(
        data=np.ones((1, 8, 12), dtype=np.float32),
        feature_type="stft",
        sample_rate=16000,
        hop_length=256,
    )

    with pytest.raises(ValueError, match="two-dimensional map data"):
        prepare_multi_curve_preview(feature, _settings())


def test_singleton_feature_map_axis_is_line_compatible():
    feature = FeatureData(
        data=np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32),
        feature_type="pitch",
        sample_rate=100,
        hop_length=10,
    )

    preview = prepare_multi_curve_preview(feature, _settings(x_alignment="original"))

    assert len(preview.series) == 1
    np.testing.assert_allclose(preview.series[0].x, [0.0, 0.1, 0.2])
    np.testing.assert_allclose(preview.series[0].y, [1.0, 2.0, 3.0])


def test_channel_modes_separate_merge_and_select():
    audio = AudioData(
        data=np.array([[0.0, 2.0], [2.0, 4.0]], dtype=np.float32),
        sample_rate=2,
        file_path="stereo.wav",
    )

    separate = prepare_multi_curve_preview(audio, _settings(channel_mode="separate"))
    merged = prepare_multi_curve_preview(audio, _settings(channel_mode="merge"))
    selected = prepare_multi_curve_preview(
        audio,
        _settings(channel_mode="selected", selected_channel=1),
    )

    assert [series.name for series in separate.series] == [
        "stereo.wav · channel 1",
        "stereo.wav · channel 2",
    ]
    np.testing.assert_allclose(merged.series[0].y, [1.0, 3.0])
    np.testing.assert_allclose(selected.series[0].y, [2.0, 4.0])


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("mean_center", [-1.0, 0.0, 1.0]),
        ("minmax", [0.0, 0.5, 1.0]),
        ("zscore", [-1.22474487, 0.0, 1.22474487]),
    ],
)
def test_value_transforms(mode, expected):
    preview = prepare_multi_curve_preview(
        np.array([1.0, 2.0, 3.0]),
        _settings(value_transform=mode),
    )

    np.testing.assert_allclose(preview.series[0].y, expected)


def test_original_coordinates_disable_incompatible_aggregate():
    preview = prepare_multi_curve_preview(
        [np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0])],
        _settings(x_alignment="original"),
    )

    assert preview.aggregate_y is None
    assert any("compatible horizontal coordinates" in warning for warning in preview.warnings)


def test_median_and_percentile_interval_are_calculated_per_position():
    preview = prepare_multi_curve_preview(
        [
            np.array([0.0, 10.0]),
            np.array([2.0, 20.0]),
            np.array([100.0, 30.0]),
        ],
        _settings(
            aggregate_mode="median",
            variability_mode="percentile",
            percentile_low=25.0,
            percentile_high=75.0,
        ),
    )

    np.testing.assert_allclose(preview.aggregate_y, [2.0, 20.0])
    np.testing.assert_allclose(preview.lower_y, [1.0, 15.0])
    np.testing.assert_allclose(preview.upper_y, [51.0, 25.0])


def test_render_downsampling_preserves_every_series():
    inputs = [np.linspace(index, index + 1.0, 100) for index in range(7)]

    preview = prepare_multi_curve_preview(inputs, _settings(max_points=10))

    assert len(preview.series) == 7
    assert all(len(series.y) == 10 for series in preview.series)
    assert preview.downsampled is True


def test_node_caches_preview_without_creating_workflow_output():
    node = create_node("multi_curve_viewer")
    node.inputs["data"].data = [np.array([1.0, 2.0]), np.array([2.0, 3.0])]

    assert node.execute()
    assert node.outputs == {}
    preview_outputs = node.get_preview_outputs()
    assert list(preview_outputs) == ["curves"]
    assert isinstance(preview_outputs["curves"], MultiCurvePreviewData)

    node.reset()
    assert node.get_preview_outputs() == {}


def test_node_configuration_round_trips_through_serialization():
    node = create_node("multi_curve_viewer")
    node.set_parameter("display_mode", "offset")
    node.set_parameter("interaction_mode", "series_toggle")
    node.set_parameter("aggregate_mode", "median")
    node.set_parameter("line_opacity", 0.55)

    restored = type(node).from_dict(node.to_dict())

    assert restored.get_parameter("display_mode") == "offset"
    assert restored.get_parameter("interaction_mode") == "series_toggle"
    assert restored.get_parameter("aggregate_mode") == "median"
    assert restored.get_parameter("line_opacity") == pytest.approx(0.55)
    assert restored.outputs == {}


def test_percentile_validation_requires_ordered_bounds():
    node = create_node("multi_curve_viewer")
    node.inputs["data"].set_connected(True)
    node.set_parameter("percentile_low", 90.0)
    node.set_parameter("percentile_high", 10.0)

    valid, message = node.validate()

    assert not valid
    assert "smaller" in message


def test_preview_view_routes_payload_without_item_pagination():
    app = QApplication.instance() or QApplication([])
    payload = prepare_multi_curve_preview(
        [np.array([1.0, 2.0]), np.array([2.0, 3.0])],
        _settings(),
    )
    view = PreviewView()

    try:
        view.set_node_data("viewer_1", "Multi-curve viewer", {"curves": payload})

        assert view._stack.currentWidget() is view._preview_widgets["multi_curve"]
        assert isinstance(view._stack.currentWidget(), MultiCurvePreviewWidget)
        assert not view._item_combo.isVisible()
        assert view._port_combo.isHidden()
        assert view._refresh_btn.isHidden()
        widget = view._preview_widgets["multi_curve"]
        assert widget._series_panel.isHidden()
        assert widget._series_list.count() == 0
        assert len(widget._curve_items) <= 10
        assert widget._series_curve_items == []
        assert "curves=2" in view._dim_info.text()
    finally:
        view.close()
        app.processEvents()


def test_property_panel_uses_readable_choices_and_contextual_controls():
    app = QApplication.instance() or QApplication([])
    node = create_node("multi_curve_viewer")
    panel = PropertyPanel()

    try:
        panel.set_node(node, node.node_id)

        display_combo = panel._widgets["display_mode"]
        interaction_combo = panel._widgets["interaction_mode"]
        assert interaction_combo.itemText(0) == "High performance"
        assert display_combo.itemText(0) == "Overlay"
        assert panel._widgets["selected_channel"].isHidden()
        assert panel._widgets["percentile_low"].isHidden()
        assert panel._widgets["show_legend"].isHidden()

        interaction_combo.setCurrentIndex(
            interaction_combo.findData("series_toggle")
        )
        channel_combo = panel._widgets["channel_mode"]
        channel_combo.setCurrentIndex(channel_combo.findData("selected"))
        variability_combo = panel._widgets["variability_mode"]
        variability_combo.setCurrentIndex(variability_combo.findData("percentile"))

        assert not panel._widgets["selected_channel"].isHidden()
        assert not panel._widgets["percentile_low"].isHidden()
        assert not panel._widgets["percentile_high"].isHidden()
        assert not panel._widgets["show_legend"].isHidden()
    finally:
        panel.close()
        app.processEvents()


def test_series_selection_mode_toggles_cached_curve_without_redraw():
    app = QApplication.instance() or QApplication([])
    payload = prepare_multi_curve_preview(
        [np.array([1.0, 2.0]), np.array([2.0, 3.0])],
        _settings(interaction_mode="series_toggle"),
    )
    widget = MultiCurvePreviewWidget()

    try:
        assert widget.set_data(payload)
        assert not widget._series_panel.isHidden()
        assert widget._series_list.count() == 2
        original_curves = list(widget._series_curve_items)

        widget._series_list.item(0).setCheckState(Qt.CheckState.Unchecked)
        app.processEvents()

        assert widget._series_curve_items == original_curves
        assert widget._series_curve_items[0].isVisible() is False
        assert widget._series_curve_items[1].isVisible() is True
    finally:
        widget.close()
        app.processEvents()


def _export_item(name, x, y, *, statistic=False):
    return CurveExportItem(
        label=name or "unnamed",
        export_name=name,
        x=np.asarray(x, dtype=np.float32),
        y=np.asarray(y, dtype=np.float32),
        is_statistic=statistic,
    )


def test_curve_csv_uses_shared_x_and_preserves_empty_duplicate_names(tmp_path):
    path = tmp_path / "shared.csv"
    curves = [
        _export_item("", [0.0, 1.0], [1.0, 2.0]),
        _export_item("duplicate", [0.0, 1.0], [3.0, 4.0]),
        _export_item("duplicate", [0.0, 1.0], [5.0, 6.0]),
    ]

    write_curve_csv(path, curves)

    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.reader(file))
    assert rows == [
        ["x", "", "duplicate", "duplicate"],
        ["0.0", "1.0", "3.0", "5.0"],
        ["1.0", "2.0", "4.0", "6.0"],
    ]


def test_curve_csv_uses_xy_pairs_and_pads_shorter_curves(tmp_path):
    path = tmp_path / "independent.csv"
    curves = [
        _export_item("mean", [0.0, 1.0], [1.0, 2.0]),
        _export_item("记录", [0.0, 0.5, 1.0], [3.0, 4.0, 5.0]),
    ]

    write_curve_csv(path, curves)

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.reader(file))
    assert rows == [
        ["mean_x", "mean_y", "记录_x", "记录_y"],
        ["0.0", "1.0", "0.0", "3.0"],
        ["1.0", "2.0", "0.5", "4.0"],
        ["", "", "1.0", "5.0"],
    ]


@pytest.mark.parametrize(
    "curve",
    [
        _export_item("empty", [], []),
        _export_item("mismatch", [0.0, 1.0], [1.0]),
    ],
)
def test_curve_csv_rejects_invalid_preview_arrays_before_writing(tmp_path, curve):
    path = tmp_path / "invalid.csv"

    with pytest.raises(ValueError):
        write_curve_csv(path, [curve])

    assert not path.exists()


def test_export_dialog_defaults_empty_and_tracks_tristate_select_all():
    app = QApplication.instance() or QApplication([])
    dialog = MultiCurveExportDialog(
        [
            _export_item("mean", [0.0], [1.0], statistic=True),
            _export_item("record", [0.0], [2.0]),
        ]
    )

    try:
        assert Styles.CHECKBOX_CONTROLS in dialog.styleSheet()
        assert "border: 2px solid #7f849c" in dialog.styleSheet()
        assert "QCheckBox::indicator:hover" in dialog.styleSheet()
        assert "QCheckBox::indicator:checked" in dialog.styleSheet()
        assert "QCheckBox::indicator:indeterminate" in dialog.styleSheet()
        assert "QCheckBox::indicator:disabled" in dialog.styleSheet()
        assert dialog._select_all_checkbox.checkState() == Qt.CheckState.Unchecked
        assert all(not checkbox.isChecked() for checkbox in dialog._row_checkboxes)
        assert not dialog._confirm_button.isEnabled()

        dialog._row_checkboxes[0].setChecked(True)
        app.processEvents()
        assert dialog._select_all_checkbox.checkState() == Qt.CheckState.PartiallyChecked
        assert dialog._confirm_button.isEnabled()

        dialog._select_all_checkbox.click()
        app.processEvents()
        assert all(checkbox.isChecked() for checkbox in dialog._row_checkboxes)
        assert dialog._select_all_checkbox.checkState() == Qt.CheckState.Checked

        dialog._select_all_checkbox.click()
        app.processEvents()
        assert all(not checkbox.isChecked() for checkbox in dialog._row_checkboxes)
        assert dialog._select_all_checkbox.checkState() == Qt.CheckState.Unchecked
    finally:
        dialog.close()
        app.processEvents()


def test_export_dialog_returns_selected_curves_with_temporary_names():
    app = QApplication.instance() or QApplication([])
    dialog = MultiCurveExportDialog(
        [
            _export_item("mean", [0.0], [1.0], statistic=True),
            _export_item("record", [0.0], [2.0]),
        ]
    )

    try:
        dialog._row_checkboxes[1].setChecked(True)
        dialog._name_edits[1].setText("")

        selected = dialog.selected_curves()

        assert len(selected) == 1
        assert selected[0].export_name == ""
        np.testing.assert_allclose(selected[0].y, [2.0])
    finally:
        dialog.close()
        app.processEvents()


def test_preview_export_candidates_follow_visible_state_and_statistics_order():
    app = QApplication.instance() or QApplication([])
    payload = prepare_multi_curve_preview(
        [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
        _settings(
            interaction_mode="series_toggle",
            aggregate_mode="median",
            variability_mode="minmax",
        ),
    )
    widget = MultiCurvePreviewWidget()

    try:
        assert widget.set_data(payload)
        assert widget._export_csv_button.isEnabled()
        widget._series_list.item(0).setCheckState(Qt.CheckState.Unchecked)
        widget._filter_series("no matching curve")

        candidates = widget._build_export_candidates()

        assert [item.export_name for item in candidates[:3]] == [
            "median",
            "lowerbound",
            "upperbound",
        ]
        assert all(item.is_statistic for item in candidates[:3])
        assert len(candidates) == 4
        assert candidates[-1].export_name == payload.series[1].name
        np.testing.assert_array_equal(candidates[-1].y, payload.series[1].y)
    finally:
        widget.close()
        app.processEvents()


def test_performance_mode_exports_only_available_statistics():
    app = QApplication.instance() or QApplication([])
    payload = prepare_multi_curve_preview(
        [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
        _settings(interaction_mode="performance"),
    )
    widget = MultiCurvePreviewWidget()

    try:
        assert widget.set_data(payload)
        assert widget._export_csv_button.isEnabled()
        assert "summary curves only" in widget._export_csv_button.toolTip().lower()
        candidates = widget._build_export_candidates()
        assert [item.export_name for item in candidates] == [
            "mean",
            "lowerbound",
            "upperbound",
        ]
        assert all(item.is_statistic for item in candidates)
    finally:
        widget.close()
        app.processEvents()


@pytest.mark.parametrize(
    "settings",
    [
        _settings(interaction_mode="performance", aggregate_mode="none"),
        _settings(interaction_mode="performance", x_alignment="original"),
    ],
)
def test_performance_export_is_disabled_without_available_statistics(settings):
    app = QApplication.instance() or QApplication([])
    payload = prepare_multi_curve_preview(
        [np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0])],
        settings,
    )
    widget = MultiCurvePreviewWidget()

    try:
        assert widget.set_data(payload)
        assert not widget._export_csv_button.isEnabled()
        assert widget._build_export_candidates() == []
        assert "no summary curves" in widget._export_csv_button.toolTip().lower()
    finally:
        widget.close()
        app.processEvents()


def test_preview_export_cancel_stops_before_file_selection(monkeypatch):
    app = QApplication.instance() or QApplication([])
    payload = prepare_multi_curve_preview(
        [np.array([1.0, 2.0])],
        _settings(interaction_mode="series_toggle"),
    )
    widget = MultiCurvePreviewWidget()

    class RejectedDialog:
        def __init__(self, candidates, parent):
            assert candidates
            assert parent is widget

        def exec(self):
            return QDialog.DialogCode.Rejected

    try:
        assert widget.set_data(payload)
        monkeypatch.setattr(
            multi_curve_preview_module,
            "MultiCurveExportDialog",
            RejectedDialog,
        )
        monkeypatch.setattr(
            multi_curve_preview_module.QFileDialog,
            "getSaveFileName",
            lambda *args, **kwargs: pytest.fail("Save dialog should not open"),
        )

        widget._export_csv()
    finally:
        widget.close()
        app.processEvents()


def test_preview_export_reports_write_failure_without_success(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    payload = prepare_multi_curve_preview(
        [np.array([1.0, 2.0])],
        _settings(interaction_mode="series_toggle"),
    )
    widget = MultiCurvePreviewWidget()
    critical_messages = []
    success_messages = []

    class AcceptedDialog:
        def __init__(self, candidates, parent):
            self._selected = [candidates[0]]

        def exec(self):
            return QDialog.DialogCode.Accepted

        def selected_curves(self):
            return self._selected

    try:
        assert widget.set_data(payload)
        monkeypatch.setattr(
            multi_curve_preview_module,
            "MultiCurveExportDialog",
            AcceptedDialog,
        )
        monkeypatch.setattr(
            multi_curve_preview_module.QFileDialog,
            "getSaveFileName",
            lambda *args, **kwargs: (str(tmp_path / "curves"), ""),
        )
        monkeypatch.setattr(
            multi_curve_preview_module,
            "write_curve_csv",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")),
        )
        monkeypatch.setattr(
            multi_curve_preview_module.QMessageBox,
            "critical",
            lambda *args: critical_messages.append(args[-1]),
        )
        monkeypatch.setattr(
            multi_curve_preview_module.QMessageBox,
            "information",
            lambda *args: success_messages.append(args[-1]),
        )

        widget._export_csv()

        assert critical_messages and "disk full" in critical_messages[0]
        assert success_messages == []
    finally:
        widget.close()
        app.processEvents()


def test_preview_export_appends_csv_extension_and_reports_success(
    monkeypatch, tmp_path
):
    app = QApplication.instance() or QApplication([])
    payload = prepare_multi_curve_preview(
        [np.array([1.0, 2.0])],
        _settings(interaction_mode="series_toggle"),
    )
    widget = MultiCurvePreviewWidget()
    writes = []
    success_messages = []

    class AcceptedDialog:
        def __init__(self, candidates, parent):
            self._selected = [candidates[-1]]

        def exec(self):
            return QDialog.DialogCode.Accepted

        def selected_curves(self):
            return self._selected

    try:
        assert widget.set_data(payload)
        monkeypatch.setattr(
            multi_curve_preview_module,
            "MultiCurveExportDialog",
            AcceptedDialog,
        )
        monkeypatch.setattr(
            multi_curve_preview_module.QFileDialog,
            "getSaveFileName",
            lambda *args, **kwargs: (str(tmp_path / "curves"), ""),
        )
        monkeypatch.setattr(
            multi_curve_preview_module,
            "write_curve_csv",
            lambda path, curves: writes.append((path, curves)),
        )
        monkeypatch.setattr(
            multi_curve_preview_module.QMessageBox,
            "information",
            lambda *args: success_messages.append(args[-1]),
        )

        widget._export_csv()

        assert len(writes) == 1
        assert writes[0][0].endswith("curves.csv")
        assert len(writes[0][1]) == 1
        assert success_messages and "curves.csv" in success_messages[0]
    finally:
        widget.close()
        app.processEvents()
