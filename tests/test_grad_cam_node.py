import numpy as np
import pytest

from src.ui.i18n import tr_
from src.workflow.node_base import NodeCategory, create_node
from src.workflow.nodes.feature import FeatureData
from src.workflow.nodes.feature.grad_cam import GradCAMNode


class Dense:
    name = "dense"
    output_shape = (None, 3)


class Conv1D:
    def __init__(self, name="conv1d"):
        self.name = name
        self.output_shape = (None, 16, 8)


class Conv2D:
    def __init__(self, name="conv2d"):
        self.name = name
        self.output_shape = (None, 8, 10, 4)


class FakeModel:
    def __init__(self, layers):
        self.layers = layers

    def get_layer(self, name):
        for layer in self.layers:
            if layer.name == name:
                return layer
        raise ValueError(name)


def test_grad_cam_node_is_registered_under_model_explanation():
    node = create_node("grad_cam")

    assert node is not None
    assert node.category == NodeCategory.TRAINING
    assert node.subcategory == tr_("Model explanation")
    assert node.subcategory_order == 40
    assert node.palette_order == 10
    assert "model" in node.inputs
    assert "input_data" in node.inputs
    assert node.inputs["target"].required is False
    assert "heatmap" in node.outputs
    assert "prediction" in node.outputs
    assert "overlay" in node.outputs
    assert node.get_parameter("overlay_enabled") is False


def test_select_layer_auto_uses_last_supported_conv_layer():
    node = GradCAMNode()
    conv1 = Conv1D("early_conv")
    conv2 = Conv2D("late_conv")
    model = FakeModel([conv1, Dense(), conv2, Dense()])

    selected = node._select_layer(model)

    assert selected is conv2


def test_select_layer_validates_named_layer_is_supported():
    node = GradCAMNode()
    node.set_parameter("layer_name", "dense")

    with pytest.raises(ValueError, match="supported Conv1D/Conv2D"):
        node._select_layer(FakeModel([Conv1D(), Dense()]))


def test_select_layer_reports_when_model_has_no_supported_conv_layer():
    node = GradCAMNode()

    with pytest.raises(ValueError, match="No Conv1D or Conv2D layer"):
        node._select_layer(FakeModel([Dense()]))


def test_coerce_target_indices_broadcasts_scalar_and_argmaxes_one_hot_targets():
    node = GradCAMNode()

    scalar = node._coerce_target_indices(2, batch_size=3)
    one_hot = node._coerce_target_indices([[0.1, 0.9], [0.8, 0.2]], batch_size=2)

    assert scalar.tolist() == [2, 2, 2]
    assert one_hot.tolist() == [1, 0]


def test_coerce_target_indices_rejects_batch_size_mismatch():
    node = GradCAMNode()

    with pytest.raises(ValueError, match="batch size"):
        node._coerce_target_indices([0, 1, 2], batch_size=2)


def test_normalize_heatmaps_scales_each_sample_independently():
    node = GradCAMNode()
    heatmaps = np.array(
        [
            [[0.0, 2.0], [1.0, 4.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ],
        dtype=np.float32,
    )

    normalized = node._normalize_heatmaps(heatmaps)

    assert normalized[0].max() == pytest.approx(1.0)
    assert normalized[0, 0, 1] == pytest.approx(0.5)
    assert normalized[1].max() == pytest.approx(0.0)


def test_wrap_conv1d_heatmap_as_feature_data_uses_2d_preview_shape():
    node = GradCAMNode()
    meta = [{"sample_rate": 16000, "hop_length": 128, "source_file": "a.wav"}]
    heatmaps = np.array([[0.1, 0.5, 1.0]], dtype=np.float32)

    feature = node._wrap_heatmaps_as_feature_data(
        heatmaps,
        meta,
        is_batch=False,
        heatmap_kind="conv1d",
    )

    assert isinstance(feature, FeatureData)
    assert feature.feature_type == "grad_cam"
    assert feature.data.shape == (1, 1, 3)
    assert feature.sample_rate == 16000
    assert feature.hop_length == 128
    assert feature.source_file == "a.wav"


def test_wrap_conv2d_heatmap_as_feature_data_uses_single_aggregate_channel():
    node = GradCAMNode()
    meta = [{"sample_rate": 48000, "hop_length": 512, "source_file": "feature.npy"}]
    heatmaps = np.arange(12, dtype=np.float32).reshape(1, 3, 4)

    feature = node._wrap_heatmaps_as_feature_data(
        heatmaps,
        meta,
        is_batch=False,
        heatmap_kind="conv2d",
    )

    assert feature.data.shape == (1, 3, 4)
    assert np.array_equal(feature.data[0], heatmaps[0])


def test_adjust_data_format_converts_channels_first_feature_input_for_keras(monkeypatch):
    node = GradCAMNode()
    monkeypatch.setattr(node, "_is_keras_model", lambda model: True)
    x = np.zeros((2, 3, 8, 10), dtype=np.float32)

    converted = node._adjust_data_format(x, model=object())

    assert converted.shape == (2, 8, 10, 3)


def test_resize_conv1d_heatmap_to_input_time_length():
    node = GradCAMNode()
    heatmaps = np.array([[0.0, 1.0]], dtype=np.float32)
    meta = [{"input_shape": (2, 5)}]

    resized = node._resize_heatmaps_to_input(heatmaps, "conv1d", meta)

    assert resized.shape == (1, 5)
    assert resized[0, 0] == pytest.approx(0.0)
    assert resized[0, -1] == pytest.approx(1.0)


def test_overlay_uses_mean_channel_base_for_multichannel_feature_data():
    node = GradCAMNode()
    node.set_parameter("overlay_enabled", True)
    node.set_parameter("overlay_alpha", 0.5)
    feature = FeatureData(
        data=np.array(
            [
                [[0.0, 1.0], [2.0, 3.0]],
                [[4.0, 5.0], [6.0, 7.0]],
            ],
            dtype=np.float32,
        ),
        feature_type="mel_spectrogram",
        sample_rate=48000,
        hop_length=256,
        source_file="multi.wav",
    )
    heatmaps = np.array([[[0.0, 1.0], [1.0, 0.0]]], dtype=np.float32)
    meta = [{"sample_rate": 48000, "hop_length": 256, "source_file": "multi.wav"}]

    overlay = node._build_overlay_output(
        input_data=feature,
        heatmaps=heatmaps,
        meta_list=meta,
        is_batch=False,
        heatmap_kind="conv2d",
    )

    assert isinstance(overlay, FeatureData)
    assert overlay.feature_type == "grad_cam_overlay"
    assert overlay.data.shape == (1, 2, 2)
    assert overlay.sample_rate == 48000
    assert overlay.hop_length == 256
    assert np.all(overlay.data >= 0.0)
    assert np.all(overlay.data <= 1.0)


def test_overlay_first_channel_mode_uses_first_source_channel():
    node = GradCAMNode()
    node.set_parameter("overlay_channel_mode", "first")
    feature = FeatureData(
        data=np.array(
            [
                [[0.0, 10.0], [20.0, 30.0]],
                [[100.0, 100.0], [100.0, 100.0]],
            ],
            dtype=np.float32,
        ),
        feature_type="mel_spectrogram",
        sample_rate=48000,
        hop_length=512,
    )

    base = node._extract_overlay_base(feature)

    assert np.array_equal(base, feature.data[0])


def test_overlay_expands_conv1d_heatmap_across_feature_rows():
    node = GradCAMNode()
    node.set_parameter("overlay_mode", "blend")
    node.set_parameter("overlay_alpha", 1.0)
    base = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32)
    heatmap = np.array([0.0, 1.0], dtype=np.float32)

    overlay = node._blend_overlay_matrix(base, heatmap, "conv1d")

    assert overlay.shape == base.shape
    assert overlay[0, 0] == pytest.approx(0.0)
    assert overlay[0, -1] == pytest.approx(1.0)
    assert np.array_equal(overlay[0], overlay[1])


def test_overlay_mask_mode_uses_cam_as_source_visibility_mask():
    node = GradCAMNode()
    node.set_parameter("overlay_mode", "mask")
    node.set_parameter("overlay_alpha", 1.0)
    base = np.array([[0.0, 5.0, 10.0]], dtype=np.float32)
    heatmap = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)

    overlay = node._blend_overlay_matrix(base, heatmap, "conv2d")

    assert overlay.shape == base.shape
    assert overlay[0, 0] == pytest.approx(0.0)
    assert overlay[0, 1] == pytest.approx(0.25)
    assert overlay[0, 2] == pytest.approx(1.0)


def test_overlay_mask_mode_alpha_preserves_context_in_low_cam_regions():
    node = GradCAMNode()
    node.set_parameter("overlay_mode", "mask")
    node.set_parameter("overlay_alpha", 0.5)
    base = np.array([[0.0, 5.0, 10.0]], dtype=np.float32)
    heatmap = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

    overlay = node._blend_overlay_matrix(base, heatmap, "conv2d")

    assert overlay[0, 0] == pytest.approx(0.0)
    assert overlay[0, 1] == pytest.approx(0.25)
    assert overlay[0, 2] == pytest.approx(0.5)


def test_compute_grad_cam_with_tiny_conv2d_model():
    tf = pytest.importorskip("tensorflow")
    from tensorflow import keras

    node = GradCAMNode()
    inputs = keras.Input(shape=(6, 8, 1))
    x = keras.layers.Conv2D(2, 3, activation="relu", name="target_conv")(inputs)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(2, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    sample = np.ones((1, 6, 8, 1), dtype=np.float32)

    heatmaps, predictions, heatmap_kind = node._compute_grad_cam(
        model=model,
        layer=model.get_layer("target_conv"),
        x=sample,
        target_data=None,
    )

    assert heatmap_kind == "conv2d"
    assert heatmaps.shape == (1, 4, 6)
    assert predictions.shape == (1, 2)
    assert np.all(heatmaps >= 0)


def test_compute_grad_cam_rejects_out_of_range_fixed_target_class():
    tf = pytest.importorskip("tensorflow")
    from tensorflow import keras

    node = GradCAMNode()
    node.set_parameter("target_mode", "fixed")
    node.set_parameter("target_class", 5)
    inputs = keras.Input(shape=(6, 8, 1))
    x = keras.layers.Conv2D(2, 3, activation="relu", name="target_conv")(inputs)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(2, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    with pytest.raises(ValueError, match="out of range"):
        node._compute_grad_cam(
            model=model,
            layer=model.get_layer("target_conv"),
            x=np.ones((1, 6, 8, 1), dtype=np.float32),
            target_data=None,
        )
