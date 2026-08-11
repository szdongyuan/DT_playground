# -*- coding: utf-8 -*-
"""
Grad-CAM feature explainability node.

The node generates sample-level Grad-CAM heatmaps from Conv1D or Conv2D Keras
layers and wraps them as FeatureData so existing workflow previews can render
the result.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ...node_base import BaseNode, NodeCategory, register_node
from ...port import DataType
from .base import AudioData, FeatureData
from src.ui.i18n import tr_

logger = logging.getLogger(__name__)


@register_node
class GradCAMNode(BaseNode):
    """
    Grad-CAM explainability node.

    Inputs:
    - model: Keras model
    - input_data: AudioData/FeatureData/np.ndarray or list form
    - target: optional class index / indices

    Output:
    - heatmap: FeatureData or List[FeatureData]
    - prediction: raw model prediction
    """

    node_type = "grad_cam"
    display_name = tr_("Grad-CAM")
    category = NodeCategory.TRAINING
    subcategory = tr_("Model explanation")
    subcategory_order = 40
    palette_order = 10
    description = tr_("Generate Grad-CAM heatmaps from Conv1D or Conv2D model layers")
    icon = "🔥"

    def _setup_ports(self):
        self.add_input("model", DataType.MODEL, tr_("Model"))
        self.add_input("input_data", DataType.ANY, tr_("Input data"))
        self.add_input("target", DataType.ANY, tr_("Target class"), required=False)
        self.add_output("heatmap", DataType.FEATURE_2D, tr_("Grad-CAM heatmap"))
        self.add_output("prediction", DataType.ANY, tr_("Prediction"))
        self.add_output("overlay", DataType.FEATURE_2D, tr_("Grad-CAM overlay"))

    def _setup_parameters(self):
        self.add_parameter(
            "layer_name",
            "str",
            "",
            display_name=tr_("Layer name (optional)"),
            description=tr_("If set, use this Conv1D/Conv2D layer; otherwise auto-select the last compatible layer"),
        )
        self.add_parameter(
            "target_mode",
            "choice",
            "auto",
            display_name=tr_("Target mode"),
            choices=["auto", "fixed", "input"],
            description=tr_("auto: use predicted class; fixed: use Target class; input: use optional target port"),
        )
        self.add_parameter(
            "target_class",
            "int",
            -1,
            display_name=tr_("Target class"),
            min_value=-1,
            description=tr_("Class index for fixed mode. -1 falls back to auto."),
        )
        self.add_parameter(
            "normalize",
            "bool",
            True,
            display_name=tr_("Normalize heatmap"),
            description=tr_("Normalize each heatmap to the 0..1 range"),
        )
        self.add_parameter(
            "resize_to_input",
            "bool",
            True,
            display_name=tr_("Resize to input"),
            description=tr_("Resize the heatmap to the input feature/time dimensions when possible"),
        )
        self.add_parameter(
            "overlay_enabled",
            "bool",
            False,
            display_name=tr_("Generate overlay"),
            description=tr_("Generate a preview-compatible intensity overlay for 2D feature inputs"),
        )
        self.add_parameter(
            "overlay_alpha",
            "float",
            0.45,
            display_name=tr_("Overlay alpha"),
            min_value=0.0,
            max_value=1.0,
            description=tr_("Blend strength of the Grad-CAM heatmap in the overlay"),
        )
        self.add_parameter(
            "overlay_mode",
            "choice",
            "mask",
            display_name=tr_("Overlay mode"),
            choices=["mask", "blend"],
            description=tr_("mask: CAM controls source visibility; blend: mix source and CAM intensity"),
        )
        self.add_parameter(
            "overlay_channel_mode",
            "choice",
            "mean",
            display_name=tr_("Overlay channel mode"),
            choices=["mean", "first"],
            description=tr_("How to reduce multi-channel source features for overlay generation"),
        )
        self.add_parameter(
            "batch_size",
            "int",
            16,
            display_name=tr_("Batch size"),
            min_value=1,
        )

    def execute(self) -> bool:
        model = self.get_input_data("model")
        input_data = self.get_input_data("input_data")
        target_data = self.get_input_data("target")

        if model is None:
            self.error_message = tr_("No model provided")
            return False
        if input_data is None:
            self.error_message = tr_("No input data provided")
            return False

        try:
            x, meta_list, is_batch = self._convert_to_array_and_meta(input_data, model)
            layer = self._select_layer(model)
        except Exception as e:
            self.error_message = tr_("Failed to prepare Grad-CAM input: {error}").format(error=str(e))
            logger.exception("Grad-CAM input preparation failed")
            return False

        try:
            heatmaps, predictions, heatmap_kind = self._compute_grad_cam(
                model=model,
                layer=layer,
                x=x,
                target_data=target_data,
            )
            if self.get_parameter("resize_to_input"):
                heatmaps = self._resize_heatmaps_to_input(heatmaps, heatmap_kind, meta_list)

            wrapped = self._wrap_heatmaps_as_feature_data(
                heatmaps,
                meta_list,
                is_batch=is_batch,
                heatmap_kind=heatmap_kind,
            )
            self.set_output_data("heatmap", wrapped)
            self.set_output_data("prediction", predictions)
            if self.get_parameter("overlay_enabled"):
                overlay = self._build_overlay_output(
                    input_data=input_data,
                    heatmaps=heatmaps,
                    meta_list=meta_list,
                    is_batch=is_batch,
                    heatmap_kind=heatmap_kind,
                )
                if overlay is not None:
                    self.set_output_data("overlay", overlay)
            self.report_status(
                tr_("Grad-CAM generated from layer: {layer}").format(layer=getattr(layer, "name", ""))
            )
            return True
        except Exception as e:
            self.error_message = tr_("Grad-CAM failed: {error}").format(error=str(e))
            logger.exception("Grad-CAM execution failed")
            return False

    def _select_layer(self, model: Any) -> Any:
        layer_name = (self.get_parameter("layer_name") or "").strip()
        if layer_name:
            try:
                layer = model.get_layer(layer_name)
            except Exception as e:
                raise ValueError(tr_("Layer not found: {name}").format(name=layer_name)) from e
            if not self._is_supported_conv_layer(layer):
                raise ValueError(tr_("Layer is not a supported Conv1D/Conv2D layer: {name}").format(name=layer_name))
            return layer

        layers = list(getattr(model, "layers", []) or [])
        for layer in reversed(layers):
            if self._is_supported_conv_layer(layer):
                return layer
        raise ValueError(tr_("No Conv1D or Conv2D layer found for Grad-CAM"))

    def _is_supported_conv_layer(self, layer: Any) -> bool:
        class_name = layer.__class__.__name__
        if class_name in {"Conv1D", "Conv2D"}:
            return True
        output_shape = getattr(layer, "output_shape", None)
        if output_shape is None:
            output = getattr(layer, "output", None)
            output_shape = getattr(output, "shape", None)
        if output_shape is None:
            return False
        try:
            rank = len(output_shape)
        except TypeError:
            rank = getattr(output_shape, "rank", None)
        return rank in (3, 4) and "conv" in class_name.lower()

    def _compute_grad_cam(
        self,
        *,
        model: Any,
        layer: Any,
        x: np.ndarray,
        target_data: Any,
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        try:
            import tensorflow as tf  # type: ignore
            from tensorflow import keras  # type: ignore
        except Exception as e:
            raise RuntimeError(tr_("TensorFlow/Keras is required: {error}").format(error=str(e))) from e

        grad_model = keras.Model(inputs=model.inputs, outputs=[layer.output, model.output])
        x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(x_tensor, training=False)
            if isinstance(predictions, (list, tuple)):
                raise ValueError(tr_("Grad-CAM requires a single model output"))
            target_indices = self._resolve_target_indices(predictions, target_data)
            scores = self._gather_target_scores(predictions, target_indices)

        gradients = tape.gradient(scores, conv_outputs)
        if gradients is None:
            raise ValueError(tr_("Could not compute gradients for the selected layer"))

        rank = len(conv_outputs.shape)
        if rank == 3:
            weights = tf.reduce_mean(gradients, axis=1)
            cam = tf.reduce_sum(conv_outputs * weights[:, tf.newaxis, :], axis=-1)
            heatmap_kind = "conv1d"
        elif rank == 4:
            weights = tf.reduce_mean(gradients, axis=(1, 2))
            cam = tf.reduce_sum(conv_outputs * weights[:, tf.newaxis, tf.newaxis, :], axis=-1)
            heatmap_kind = "conv2d"
        else:
            raise ValueError(tr_("Selected layer output rank is not supported for Grad-CAM"))

        cam = tf.nn.relu(cam)
        heatmaps = cam.numpy().astype(np.float32)
        if self.get_parameter("normalize"):
            heatmaps = self._normalize_heatmaps(heatmaps)

        return heatmaps, predictions.numpy(), heatmap_kind

    def _resolve_target_indices(self, predictions: Any, target_data: Any) -> Any:
        import tensorflow as tf  # type: ignore

        pred_shape = predictions.shape
        batch_size = int(pred_shape[0])
        output_dim = int(pred_shape[-1]) if len(pred_shape) > 1 and pred_shape[-1] is not None else 1
        target_mode = self.get_parameter("target_mode") or "auto"

        if target_mode == "input":
            indices = self._coerce_target_indices(target_data, batch_size)
            if indices is not None:
                return tf.convert_to_tensor(indices, dtype=tf.int32)

        if target_mode == "fixed":
            fixed_class = int(self.get_parameter("target_class"))
            if fixed_class >= 0:
                return tf.fill([batch_size], fixed_class)

        if output_dim <= 1:
            return tf.zeros([batch_size], dtype=tf.int32)
        return tf.argmax(predictions, axis=-1, output_type=tf.int32)

    def _coerce_target_indices(self, target_data: Any, batch_size: int) -> Optional[np.ndarray]:
        if target_data is None:
            return None
        arr = np.asarray(target_data)
        if arr.size == 0:
            return None
        if arr.ndim == 0:
            return np.full((batch_size,), int(arr), dtype=np.int32)
        if arr.ndim > 1:
            arr = np.argmax(arr, axis=-1)
        arr = arr.astype(np.int32).reshape(-1)
        if arr.size == 1 and batch_size > 1:
            return np.full((batch_size,), int(arr[0]), dtype=np.int32)
        if arr.size != batch_size:
            raise ValueError(tr_("Target input size does not match batch size"))
        return arr

    def _gather_target_scores(self, predictions: Any, target_indices: Any) -> Any:
        import tensorflow as tf  # type: ignore

        if len(predictions.shape) == 1:
            return predictions
        if int(predictions.shape[-1]) == 1:
            return predictions[:, 0]
        output_dim = int(predictions.shape[-1])
        try:
            target_array = target_indices.numpy() if hasattr(target_indices, "numpy") else np.asarray(target_indices)
            if np.any(target_array < 0) or np.any(target_array >= output_dim):
                raise ValueError(
                    tr_("Target class index is out of range for model output size {size}").format(
                        size=output_dim
                    )
                )
        except ValueError:
            raise
        row_indices = tf.range(tf.shape(predictions)[0], dtype=tf.int32)
        gather_indices = tf.stack([row_indices, target_indices], axis=1)
        return tf.gather_nd(predictions, gather_indices)

    def _convert_to_array_and_meta(
        self, data: Any, model: Any
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], bool]:
        is_batch = isinstance(data, list)
        items: List[Any] = data if isinstance(data, list) else [data]
        if len(items) == 0:
            raise ValueError(tr_("Input data list is empty"))

        meta_list = [self._extract_meta(item) for item in items]

        if isinstance(items[0], AudioData):
            x = np.array([item.data for item in items], dtype=np.float32)
        elif isinstance(items[0], FeatureData):
            x = np.array([item.data for item in items], dtype=np.float32)
        else:
            x = np.array(items, dtype=np.float32)
            if x.dtype == object:
                raise ValueError(tr_("Input samples have inconsistent shapes; cannot stack into a batch array"))

        x = self._adjust_data_format(x, model)
        return x, meta_list, is_batch

    def _extract_meta(self, item: Any) -> Dict[str, Any]:
        if isinstance(item, AudioData):
            data_shape = tuple(getattr(item, "data", np.array([])).shape)
            return {
                "sample_rate": int(getattr(item, "sample_rate", 48000) or 48000),
                "hop_length": 512,
                "source_file": getattr(item, "file_path", "") or "",
                "input_shape": data_shape,
                "input_kind": "audio",
            }
        if isinstance(item, FeatureData):
            data_shape = tuple(getattr(item, "data", np.array([])).shape)
            return {
                "sample_rate": int(getattr(item, "sample_rate", 48000) or 48000),
                "hop_length": int(getattr(item, "hop_length", 512) or 512),
                "source_file": getattr(item, "source_file", "") or "",
                "input_shape": data_shape,
                "input_kind": "feature",
            }
        arr = np.asarray(item)
        return {
            "sample_rate": 48000,
            "hop_length": 512,
            "source_file": "",
            "input_shape": tuple(arr.shape),
            "input_kind": "array",
        }

    def _adjust_data_format(self, x: np.ndarray, model: Any) -> np.ndarray:
        if model is None or not self._is_keras_model(model):
            return x
        if x.ndim == 3:
            if x.shape[1] < x.shape[2]:
                return np.transpose(x, (0, 2, 1))
        elif x.ndim == 4:
            if x.shape[1] < x.shape[2] and x.shape[1] < x.shape[3]:
                return np.transpose(x, (0, 2, 3, 1))
        return x

    def _is_keras_model(self, model: Any) -> bool:
        try:
            from tensorflow import keras  # type: ignore

            return isinstance(model, keras.Model)
        except Exception:
            return False

    def _normalize_heatmaps(self, heatmaps: np.ndarray) -> np.ndarray:
        arr = np.asarray(heatmaps, dtype=np.float32)
        flat = arr.reshape(arr.shape[0], -1)
        max_values = flat.max(axis=1)
        normalized = arr.copy()
        for i, max_value in enumerate(max_values):
            if max_value > 0:
                normalized[i] = normalized[i] / max_value
        return normalized.astype(np.float32)

    def _resize_heatmaps_to_input(
        self,
        heatmaps: np.ndarray,
        heatmap_kind: str,
        meta_list: List[Dict[str, Any]],
    ) -> np.ndarray:
        resized = []
        for i, heatmap in enumerate(np.asarray(heatmaps, dtype=np.float32)):
            meta = meta_list[i] if i < len(meta_list) else meta_list[0]
            target_shape = self._target_heatmap_shape(meta, heatmap_kind)
            if target_shape is None:
                resized.append(heatmap)
            elif heatmap_kind == "conv1d":
                resized.append(self._resize_1d(heatmap, int(target_shape[0])))
            else:
                resized.append(self._resize_2d(heatmap, (int(target_shape[0]), int(target_shape[1]))))
        return np.stack(resized, axis=0).astype(np.float32)

    def _target_heatmap_shape(self, meta: Dict[str, Any], heatmap_kind: str) -> Optional[Tuple[int, ...]]:
        input_shape = tuple(meta.get("input_shape") or ())
        if heatmap_kind == "conv1d":
            if len(input_shape) >= 1:
                return (int(input_shape[-1]),)
            return None
        if len(input_shape) >= 3:
            return (int(input_shape[-2]), int(input_shape[-1]))
        if len(input_shape) == 2:
            return (int(input_shape[0]), int(input_shape[1]))
        return None

    def _resize_1d(self, values: np.ndarray, target_length: int) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        if target_length <= 0 or arr.size == target_length:
            return arr.astype(np.float32)
        if arr.size == 1:
            return np.full((target_length,), float(arr[0]), dtype=np.float32)
        source_x = np.linspace(0.0, 1.0, num=arr.size)
        target_x = np.linspace(0.0, 1.0, num=target_length)
        return np.interp(target_x, source_x, arr).astype(np.float32)

    def _resize_2d(self, values: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        target_h, target_w = target_shape
        if target_h <= 0 or target_w <= 0 or arr.shape == target_shape:
            return arr.astype(np.float32)
        try:
            import tensorflow as tf  # type: ignore

            resized = tf.image.resize(arr[np.newaxis, :, :, np.newaxis], (target_h, target_w), method="bilinear")
            return resized.numpy()[0, :, :, 0].astype(np.float32)
        except Exception:
            row_resized = np.vstack([self._resize_1d(row, target_w) for row in arr])
            columns = [self._resize_1d(row_resized[:, col], target_h) for col in range(target_w)]
            return np.stack(columns, axis=1).astype(np.float32)

    def _wrap_heatmaps_as_feature_data(
        self,
        heatmaps: np.ndarray,
        meta_list: List[Dict[str, Any]],
        *,
        is_batch: bool,
        heatmap_kind: str,
    ) -> Union[FeatureData, List[FeatureData]]:
        arr = np.asarray(heatmaps, dtype=np.float32)
        if not is_batch and arr.ndim in (1, 2):
            arr = arr.reshape((1, *arr.shape))
        if arr.shape[0] != len(meta_list):
            if len(meta_list) == 1:
                meta_list = meta_list * int(arr.shape[0])
            else:
                raise ValueError(tr_("Heatmap batch size does not match input batch size"))

        out_list = [
            self._wrap_single_heatmap(arr[i], meta_list[i], heatmap_kind=heatmap_kind)
            for i in range(arr.shape[0])
        ]
        return out_list if is_batch else out_list[0]

    def _wrap_single_heatmap(
        self,
        heatmap: np.ndarray,
        meta: Dict[str, Any],
        *,
        heatmap_kind: str,
    ) -> FeatureData:
        arr = np.asarray(heatmap, dtype=np.float32)
        if heatmap_kind == "conv1d":
            data = arr.reshape(1, 1, -1)
        else:
            data = arr.reshape(1, *arr.shape)
        return FeatureData(
            data=data.astype(np.float32),
            feature_type="grad_cam",
            sample_rate=int(meta.get("sample_rate", 48000)),
            hop_length=int(meta.get("hop_length", 512)),
            source_file=str(meta.get("source_file", "")),
        )

    def _build_overlay_output(
        self,
        *,
        input_data: Any,
        heatmaps: np.ndarray,
        meta_list: List[Dict[str, Any]],
        is_batch: bool,
        heatmap_kind: str,
    ) -> Optional[Union[FeatureData, List[FeatureData]]]:
        items: List[Any] = input_data if isinstance(input_data, list) else [input_data]
        overlays: List[FeatureData] = []

        for i, item in enumerate(items):
            if i >= len(heatmaps):
                break
            base = self._extract_overlay_base(item)
            if base is None:
                continue
            overlay_matrix = self._blend_overlay_matrix(base, heatmaps[i], heatmap_kind)
            meta = meta_list[i] if i < len(meta_list) else meta_list[0]
            overlays.append(
                FeatureData(
                    data=overlay_matrix.reshape(1, *overlay_matrix.shape).astype(np.float32),
                    feature_type="grad_cam_overlay",
                    sample_rate=int(meta.get("sample_rate", 48000)),
                    hop_length=int(meta.get("hop_length", 512)),
                    source_file=str(meta.get("source_file", "")),
                )
            )

        if not overlays:
            return None
        return overlays if is_batch else overlays[0]

    def _extract_overlay_base(self, item: Any) -> Optional[np.ndarray]:
        if isinstance(item, FeatureData):
            arr = np.asarray(item.data, dtype=np.float32)
        elif isinstance(item, AudioData):
            arr = np.asarray(item.data, dtype=np.float32)
        elif isinstance(item, np.ndarray):
            arr = np.asarray(item, dtype=np.float32)
        else:
            try:
                arr = np.asarray(item, dtype=np.float32)
            except Exception:
                return None

        if arr.ndim == 1:
            return arr.reshape(1, -1)
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3:
            mode = self.get_parameter("overlay_channel_mode") or "mean"
            if mode == "first":
                return arr[0]
            return np.mean(arr, axis=0)
        return None

    def _blend_overlay_matrix(
        self,
        base: np.ndarray,
        heatmap: np.ndarray,
        heatmap_kind: str,
    ) -> np.ndarray:
        base_2d = np.asarray(base, dtype=np.float32)
        if base_2d.ndim != 2:
            base_2d = base_2d.reshape(1, -1)

        cam = np.asarray(heatmap, dtype=np.float32)
        if heatmap_kind == "conv1d" or cam.ndim == 1:
            cam = self._resize_1d(cam, base_2d.shape[-1])
            cam_2d = np.tile(cam.reshape(1, -1), (base_2d.shape[0], 1))
        else:
            cam_2d = self._resize_2d(cam, base_2d.shape)

        base_norm = self._normalize_matrix(base_2d)
        cam_norm = self._normalize_matrix(cam_2d)
        alpha = float(self.get_parameter("overlay_alpha"))
        alpha = min(1.0, max(0.0, alpha))
        overlay_mode = self.get_parameter("overlay_mode") or "mask"
        if overlay_mode == "blend":
            return ((1.0 - alpha) * base_norm + alpha * cam_norm).astype(np.float32)

        visibility_mask = (1.0 - alpha) + alpha * cam_norm
        return (base_norm * visibility_mask).astype(np.float32)

    def _normalize_matrix(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        if arr.size == 0:
            return arr
        min_value = float(np.min(arr))
        max_value = float(np.max(arr))
        span = max_value - min_value
        if span <= 0:
            return np.zeros_like(arr, dtype=np.float32)
        return ((arr - min_value) / span).astype(np.float32)
