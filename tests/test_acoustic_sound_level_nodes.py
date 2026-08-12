import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QLineEdit

from src.audio.acoustics import (
    apply_frequency_weighting,
    exponential_sound_level,
    fractional_octave_bands,
)
from src.ui.node_editor.property_panel import PropertyPanel
from src.ui.views.preview_view import PreviewView
from src.ui.widgets.preview.curve_preview import CurvePreviewWidget
from src.ui.views.preview_view import PreviewView
from src.workflow.curve import CurveData
from src.workflow.node_base import NodeCategory, create_node
from src.workflow.nodes.data_source import AudioData
from src.workflow.nodes.feature import FeatureData
from src.workflow.nodes.visualization import prepare_multi_curve_preview
from src.workflow.port import DataType


def _tone(
    frequency=1000.0,
    rms=0.1,
    sample_rate=8192,
    duration=4.0,
    channels=1,
):
    time = np.arange(int(sample_rate * duration), dtype=np.float64) / sample_rate
    wave = np.sqrt(2.0) * rms * np.sin(2.0 * np.pi * frequency * time)
    data = np.stack([wave * (index + 1) for index in range(channels)], axis=0)
    return AudioData(data.astype(np.float32), sample_rate, "tone.wav")


def _viewer_settings(**overrides):
    settings = {
        "interaction_mode": "performance",
        "display_mode": "overlay",
        "x_alignment": "normalized",
        "frequency_axis": "auto",
        "channel_mode": "separate",
        "selected_channel": 0,
        "value_transform": "none",
        "aggregate_mode": "mean",
        "variability_mode": "none",
        "percentile_low": 25.0,
        "percentile_high": 75.0,
        "max_points": 2000,
        "line_opacity": 0.3,
        "show_legend": False,
    }
    settings.update(overrides)
    return settings


def _run_node(node_type, audio, **parameters):
    node = create_node(node_type)
    node.inputs["audio"].data = audio
    for name, value in parameters.items():
        success, message = node.set_parameter(name, value)
        assert success, message
    assert node.execute(), node.error_message
    return node.outputs["curve"].data


def test_curve_data_requires_matching_strictly_increasing_coordinates():
    with pytest.raises(ValueError, match="match the number of points"):
        CurveData(
            data=np.ones((1, 3)),
            x=np.array([0.0, 1.0]),
            x_name="Time",
            x_unit="s",
            y_name="Level",
            y_unit="dB SPL",
            curve_type="time_sliding_leq",
        )

    with pytest.raises(ValueError, match="strictly increasing"):
        CurveData(
            data=np.ones((1, 3)),
            x=np.array([0.0, 1.0, 1.0]),
            x_name="Time",
            x_unit="s",
            y_name="Level",
            y_unit="dB SPL",
            curve_type="time_sliding_leq",
        )


def test_sound_level_nodes_use_curve_ports_and_acoustic_catalog_group():
    time_node = create_node("time_varying_sound_level")
    frequency_node = create_node("steady_state_frequency_sound_level")
    converter = create_node("curve_to_feature")

    assert time_node.category == NodeCategory.FEATURE
    assert time_node.subcategory == "Acoustic analysis"
    assert frequency_node.subcategory == "Acoustic analysis"
    assert time_node.outputs["curve"].data_type == DataType.CURVE
    assert frequency_node.outputs["curve"].data_type == DataType.CURVE
    assert converter.inputs["curve"].data_type == DataType.CURVE
    assert converter.outputs["feature"].data_type == DataType.FEATURE_1D
    assert not DataType.is_compatible(DataType.CURVE, DataType.FEATURE)


def test_calibration_factor_is_required_and_has_no_default():
    node = create_node("time_varying_sound_level")
    node.inputs["audio"].set_connected(True)

    assert node.get_parameter("calibration_factor") is None
    valid, message = node.validate()

    assert not valid
    assert "Calibration factor" in message


def test_sliding_leq_matches_known_rms_and_calibration_scaling():
    audio = _tone(rms=0.1)
    first = _run_node(
        "time_varying_sound_level",
        audio,
        calibration_factor=1.0,
        integration_time_seconds=1.0,
        leq_step_seconds=0.5,
    )
    second = _run_node(
        "time_varying_sound_level",
        audio,
        calibration_factor=2.0,
        integration_time_seconds=1.0,
        leq_step_seconds=0.5,
    )

    expected = 20.0 * np.log10(0.1 / 20.0e-6)
    np.testing.assert_allclose(first.data, expected, atol=1.0e-4)
    np.testing.assert_allclose(second.data - first.data, 20.0 * np.log10(2.0), atol=1.0e-4)
    np.testing.assert_allclose(first.x[:2], [0.5, 1.0])


def test_time_varying_level_preserves_channels_and_floors_silence():
    multichannel = _run_node(
        "time_varying_sound_level",
        _tone(rms=0.05, channels=2),
        calibration_factor=1.0,
    )
    silent = _run_node(
        "time_varying_sound_level",
        AudioData(np.zeros((1, 8192), dtype=np.float32), 8192, "silent.wav"),
        calibration_factor=1.0,
        level_floor_db=-180.0,
    )

    assert multichannel.data.shape[0] == 2
    np.testing.assert_allclose(multichannel.data[1] - multichannel.data[0], 20.0 * np.log10(2.0), atol=1.0e-4)
    np.testing.assert_allclose(silent.data, -180.0)
    assert np.all(np.isfinite(silent.data))


def test_sound_level_batch_returns_one_curve_data_per_audio_record():
    first = _tone(duration=2.0)
    first.file_path = "first.wav"
    second = _tone(frequency=500.0, duration=2.0)
    second.file_path = "second.wav"
    node = create_node("time_varying_sound_level")
    node.inputs["audio"].data = [first, second]
    node.set_parameter("calibration_factor", 1.0)

    assert node.execute(), node.error_message
    output = node.outputs["curve"].data

    assert isinstance(output, list)
    assert len(output) == 2
    assert all(isinstance(item, CurveData) for item in output)
    assert [item.source_file for item in output] == ["first.wav", "second.wav"]


def test_fast_response_rises_faster_than_slow_response():
    sample_rate = 8000
    pressure = np.zeros((1, sample_rate * 2), dtype=np.float64)
    pressure[:, sample_rate:] = 0.1
    _, fast = exponential_sound_level(pressure, sample_rate, 0.125, 0.01, 20.0e-6, -200.0)
    _, slow = exponential_sound_level(pressure, sample_rate, 1.0, 0.01, 20.0e-6, -200.0)

    comparison_index = 110
    assert fast[0, comparison_index] > slow[0, comparison_index]


@pytest.mark.parametrize("weighting", ["Z", "A", "C"])
def test_frequency_weighting_produces_finite_pressure(weighting):
    audio = _tone(duration=2.0)
    weighted = apply_frequency_weighting(audio.data.astype(np.float64), audio.sample_rate, weighting)

    assert weighted.shape == audio.data.shape
    assert np.all(np.isfinite(weighted))


def test_narrowband_energy_matches_time_domain_level():
    audio = _tone(frequency=1024.0, rms=0.1)
    time_curve = _run_node(
        "time_varying_sound_level",
        audio,
        calibration_factor=1.0,
        integration_time_seconds=4.0,
    )
    spectrum = _run_node(
        "steady_state_frequency_sound_level",
        audio,
        calibration_factor=1.0,
        analysis_type="narrowband",
        spectral_estimator="fft",
        minimum_frequency_hz=1.0,
        maximum_frequency_hz=4096.0,
    )

    combined_level = 10.0 * np.log10(np.sum(np.power(10.0, spectrum.data[0] / 10.0)))
    assert combined_level == pytest.approx(time_curve.data[0, 0], abs=0.05)
    assert spectrum.x[np.argmax(spectrum.data[0])] == pytest.approx(1024.0)


@pytest.mark.parametrize("fraction", [1, 3, 6, 12])
def test_fractional_octave_centers_include_1000_hz(fraction):
    centers, lower, upper = fractional_octave_bands(fraction, 20.0, 20000.0)

    assert np.all(np.diff(centers) > 0)
    assert np.any(np.isclose(centers, 1000.0))
    assert np.all(lower < centers)
    assert np.all(centers < upper)


def test_fractional_octave_tone_peaks_at_1000_hz():
    curve = _run_node(
        "steady_state_frequency_sound_level",
        _tone(frequency=1000.0),
        calibration_factor=1.0,
        analysis_type="octave",
        octave_fraction="1/3",
        minimum_frequency_hz=100.0,
        maximum_frequency_hz=3000.0,
        welch_segment_length=8192,
    )

    assert curve.curve_type == "frequency_octave_3"
    assert curve.x[np.argmax(curve.data[0])] == pytest.approx(1000.0)


def test_fractional_octave_reports_insufficient_resolution():
    node = create_node("steady_state_frequency_sound_level")
    node.inputs["audio"].data = _tone(sample_rate=44100, duration=1.0)
    node.set_parameter("calibration_factor", 1.0)
    node.set_parameter("analysis_type", "octave")
    node.set_parameter("octave_fraction", "1/12")
    node.set_parameter("minimum_frequency_hz", 20.0)
    node.set_parameter("welch_segment_length", 4096)

    assert not node.execute()
    assert "segment length of at least" in node.error_message


def _curve(values, x=None, **metadata):
    values = np.asarray(values, dtype=np.float64)
    coordinates = np.arange(values.shape[-1], dtype=np.float64) if x is None else np.asarray(x)
    base_metadata = {
        "sample_rate": 100,
        "calibration_factor_pa_per_fs": 1.0,
        "reference_pressure_pa": 20.0e-6,
        "frequency_weighting": "Z",
        "calculation_mode": "sliding_leq",
        "integration_time_seconds": 1.0,
    }
    base_metadata.update(metadata)
    return CurveData(
        data=values,
        x=coordinates,
        x_name="Time",
        x_unit="s",
        y_name="Sound pressure level",
        y_unit="dB SPL",
        curve_type="time_sliding_leq",
        source_file="sample.wav",
        metadata=base_metadata,
    )


def test_curve_to_feature_strict_mode_preserves_full_shape():
    node = create_node("curve_to_feature")
    node.inputs["curve"].data = [
        _curve([[10.0, 20.0, 30.0]]),
        _curve([[11.0, 21.0, 31.0]]),
    ]

    assert node.execute(), node.error_message
    features = node.outputs["feature"].data
    assert all(isinstance(feature, FeatureData) for feature in features)
    assert [feature.data.shape for feature in features] == [(1, 3), (1, 3)]
    np.testing.assert_allclose(features[0].data, [[10.0, 20.0, 30.0]])
    np.testing.assert_allclose(features[0].metadata["x"], [0.0, 1.0, 2.0])
    assert features[0].metadata["y_unit"] == "dB SPL"
    assert features[0].metadata["alignment_mode"] == "strict"


def test_curve_to_feature_strict_mode_rejects_coordinate_or_metadata_mismatch():
    coordinate_node = create_node("curve_to_feature")
    coordinate_node.inputs["curve"].data = [
        _curve([[1.0, 2.0, 3.0]]),
        _curve([[1.0, 2.0, 3.0]], x=[0.0, 1.0, 2.1]),
    ]
    assert not coordinate_node.execute()
    assert "physical coordinates" in coordinate_node.error_message

    metadata_node = create_node("curve_to_feature")
    metadata_node.inputs["curve"].data = [
        _curve([[1.0, 2.0, 3.0]]),
        _curve([[1.0, 2.0, 3.0]], frequency_weighting="A"),
    ]
    assert not metadata_node.execute()
    assert "frequency_weighting" in metadata_node.error_message


def test_curve_to_feature_fixed_grid_resamples_non_octave_curves():
    node = create_node("curve_to_feature")
    node.set_parameter("alignment_mode", "fixed_grid")
    node.set_parameter("target_points", 3)
    node.inputs["curve"].data = [
        _curve([[0.0, 2.0]], x=[0.0, 2.0]),
        _curve([[0.0, 1.0, 2.0]], x=[0.0, 1.0, 2.0]),
    ]

    assert node.execute(), node.error_message
    np.testing.assert_allclose(node.outputs["feature"].data[0].data, [[0.0, 1.0, 2.0]])


def test_curve_to_feature_supports_logarithmic_fixed_grid():
    node = create_node("curve_to_feature")
    node.set_parameter("alignment_mode", "fixed_grid")
    node.set_parameter("target_points", 3)
    node.set_parameter("grid_spacing", "log")
    node.set_parameter("grid_start", 1.0)
    node.set_parameter("grid_end", 100.0)
    node.inputs["curve"].data = _curve([[0.0, 1.0, 2.0]], x=[1.0, 10.0, 100.0])

    assert node.execute(), node.error_message
    feature = node.outputs["feature"].data
    np.testing.assert_allclose(feature.metadata["x"], [1.0, 10.0, 100.0])


def test_multi_curve_viewer_preserves_curve_coordinates_and_energy_averages_spl():
    first = _curve([[0.0, 10.0, 20.0]])
    second = _curve([[10.0, 20.0, 30.0]])
    preview = prepare_multi_curve_preview([first, second], _viewer_settings())

    np.testing.assert_allclose(preview.series[0].x, first.x)
    expected = 10.0 * np.log10(np.mean(np.power(10.0, np.array([[0.0, 10.0, 20.0], [10.0, 20.0, 30.0]]) / 10.0), axis=0))
    np.testing.assert_allclose(preview.aggregate_y, expected, rtol=1.0e-6)
    assert preview.x_label == "Time (s)"
    assert preview.y_label == "Sound pressure level (dB SPL)"
    assert preview.x_scale == "linear"
    assert any("preserved" in warning for warning in preview.warnings)


def test_octave_curve_preview_uses_logarithmic_frequency_axis():
    curve = CurveData(
        data=np.array([[40.0, 50.0, 45.0]]),
        x=np.array([500.0, 1000.0, 2000.0]),
        x_name="Frequency",
        x_unit="Hz",
        y_name="Sound pressure level",
        y_unit="dB SPL",
        curve_type="frequency_octave_3",
    )

    preview = prepare_multi_curve_preview(curve, _viewer_settings(x_alignment="original"))

    assert preview.x_scale == "log"
    assert preview.x_label == "Frequency (Hz)"


def test_preview_view_routes_single_curve_data_to_physical_curve_preview():
    app = QApplication.instance() or QApplication([])
    view = PreviewView()
    curve = _curve([[10.0, 20.0, 30.0]])
    try:
        view.set_node_data("spl_1", "Time-varying total sound level", {"curve": curve})

        widget = view._preview_widgets["curve"]
        assert view._stack.currentWidget() is widget
        assert widget.current_data is curve
        assert view._item_combo.isHidden()
        assert not view._refresh_btn.isHidden()
    finally:
        view.close()
        app.processEvents()


def test_preview_view_routes_curve_batch_with_item_pagination():
    app = QApplication.instance() or QApplication([])
    view = PreviewView()
    curves = [
        _curve([[10.0, 20.0, 30.0]]),
        _curve([[11.0, 21.0, 31.0]]),
    ]
    try:
        view.set_node_data("spl_batch", "Time-varying total sound level", {"curve": curves})

        widget = view._preview_widgets["curve"]
        assert view._stack.currentWidget() is widget
        assert widget.current_data is curves[0]
        assert view._item_combo.count() == 2
        assert not view._item_combo.isHidden()
    finally:
        view.close()
        app.processEvents()


def test_property_panel_handles_unset_calibration_and_generic_dependencies():
    app = QApplication.instance() or QApplication([])
    node = create_node("steady_state_frequency_sound_level")
    panel = PropertyPanel()
    try:
        panel.set_node(node, node.node_id)
        assert isinstance(panel._widgets["calibration_factor"], QLineEdit)
        assert panel._widgets["calibration_factor"].text() == ""
        assert panel._widgets["spectral_estimator"].isHidden()
        assert not panel._widgets["octave_fraction"].isHidden()
        assert not panel._widgets["welch_segment_length"].isHidden()

        analysis = panel._widgets["analysis_type"]
        analysis.setCurrentIndex(analysis.findData("narrowband"))
        estimator = panel._widgets["spectral_estimator"]
        estimator.setCurrentIndex(estimator.findData("fft"))

        assert not estimator.isHidden()
        assert not panel._widgets["fft_size"].isHidden()
        assert panel._widgets["octave_fraction"].isHidden()
        assert panel._widgets["welch_segment_length"].isHidden()
    finally:
        panel.close()
        app.processEvents()


def test_direct_curve_preview_pages_audio_records_and_uses_source_filenames():
    app = QApplication.instance() or QApplication([])
    first = _curve([[10.0, 20.0, 30.0]])
    first.source_file = "first.wav"
    second = _curve([[11.0, 21.0, 31.0]])
    second.source_file = "second.wav"
    view = PreviewView()
    try:
        view.set_node_data("spl_1", "Time-varying total sound level", {"curve": [first, second]})

        assert view._item_combo.count() == 2
        assert view._item_combo.itemText(0) == "[0] first.wav"
        assert view._item_combo.itemText(1) == "[1] second.wav"
        assert view._stack.currentWidget() is view._preview_widgets["curve"]
        assert view._preview_widgets["curve"].current_data is first

        view._item_combo.setCurrentIndex(1)
        app.processEvents()

        assert view._preview_widgets["curve"].current_data is second
    finally:
        view.close()
        app.processEvents()


def test_curve_preview_defaults_to_channel_one_and_switches_channels():
    app = QApplication.instance() or QApplication([])
    curve = _curve([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    curve.channel_names = ["Ch 1", "Ch 2"]
    widget = CurvePreviewWidget()
    try:
        assert widget.set_data(curve)
        assert widget._channel_combo.count() == 2
        assert widget._channel_combo.currentData() == 0
        assert widget._channel_combo.currentText() == "Ch 1"
        assert "channel=Ch 1" in widget.data_info

        widget._channel_combo.setCurrentIndex(1)
        app.processEvents()

        assert widget._channel_combo.currentData() == 1
        assert "channel=Ch 2" in widget.data_info
    finally:
        widget.close()
        app.processEvents()
