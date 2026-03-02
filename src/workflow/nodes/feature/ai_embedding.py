# -*- coding: utf-8 -*-
"""
AI embedding / activation feature extraction node.

This node runs a provided Keras model on the input data and extracts the
activations from a selected layer, then wraps the result as FeatureData.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ...node_base import BaseNode, NodeCategory, register_node
from ...port import DataType
from .base import AudioData, FeatureData
from src.ui.i18n import tr_

logger = logging.getLogger(__name__)


@register_node
class AIFeatureExtractionNode(BaseNode):
    """
    AI feature extraction node (embedding/activation).

    Inputs:
    - model: Keras model
    - input_data: AudioData/FeatureData/np.ndarray or their list form

    Output:
    - feature: FeatureData or List[FeatureData]
    """

    node_type = "ai_feature_extraction"
    display_name = tr_("AI feature extraction")
    category = NodeCategory.FEATURE
    subcategory = tr_("AI features")
    description = tr_("Extract embeddings/activations from a model layer")
    icon = "🤖"

    def _setup_ports(self):
        self.add_input("model", DataType.MODEL, tr_("Model"))
        self.add_input("input_data", DataType.ANY, tr_("Input data"))
        self.add_output("feature", DataType.FEATURE, tr_("AI features"))

    def _setup_parameters(self):
        self.add_parameter(
            "layer_name",
            "str",
            "",
            display_name=tr_("Layer name (optional)"),
            description=tr_("If set, extract activations from this layer name; otherwise auto-select"),
        )
        self.add_parameter(
            "auto_layer_strategy",
            "choice",
            "last_hidden",
            display_name=tr_("Auto layer strategy"),
            choices=["last_hidden", "last_non_input", "model_output"],
            description=tr_("How to auto-select layer when layer name is empty"),
        )
        self.add_parameter(
            "batch_size",
            "int",
            32,
            display_name=tr_("Batch size"),
            min_value=1,
        )

    def execute(self) -> bool:
        model = self.get_input_data("model")
        input_data = self.get_input_data("input_data")

        if model is None:
            self.error_message = tr_("No model provided")
            return False
        if input_data is None:
            self.error_message = tr_("No input data provided")
            return False

        try:
            x, meta_list, is_batch = self._convert_to_array_and_meta(input_data, model)
        except Exception as e:
            self.error_message = tr_("Failed to prepare input data: {error}").format(error=str(e))
            logger.exception("AI feature extraction: input conversion failed")
            return False

        try:
            layer = self._select_layer(model)
        except Exception as e:
            self.error_message = tr_("Failed to select layer: {error}").format(error=str(e))
            return False

        try:
            from tensorflow import keras  # type: ignore
        except Exception as e:
            self.error_message = tr_("TensorFlow/Keras is required: {error}").format(error=str(e))
            return False

        try:
            feature_model = keras.Model(inputs=model.inputs, outputs=layer.output)
            batch_size = int(self.get_parameter("batch_size"))
            self.report_status(
                tr_("Extracting AI features from layer: {layer}").format(layer=getattr(layer, "name", ""))
            )

            y = feature_model.predict(x, batch_size=batch_size, verbose=0)
            if isinstance(y, (list, tuple)):
                raise ValueError(tr_("Selected layer produced multiple outputs; please choose a single-output layer"))

            features = self._wrap_outputs_as_feature_data(y, meta_list, is_batch=is_batch)
            self.set_output_data("feature", features)
            return True

        except Exception as e:
            self.error_message = tr_("AI feature extraction failed: {error}").format(error=str(e))
            logger.exception("AI feature extraction failed")
            return False

    def _select_layer(self, model) -> Any:
        layer_name = (self.get_parameter("layer_name") or "").strip()
        if layer_name:
            try:
                return model.get_layer(layer_name)
            except Exception as e:
                raise ValueError(tr_("Layer not found: {name}").format(name=layer_name)) from e

        strategy = self.get_parameter("auto_layer_strategy") or "last_hidden"
        layers = list(getattr(model, "layers", []) or [])
        if not layers:
            raise ValueError(tr_("Model has no layers"))

        non_input_layers = [l for l in layers if not self._is_input_layer(l)]
        if not non_input_layers:
            return layers[-1]

        if strategy == "model_output":
            return non_input_layers[-1]

        if strategy == "last_hidden":
            if len(non_input_layers) >= 2:
                return non_input_layers[-2]
            return non_input_layers[-1]

        # last_non_input
        return non_input_layers[-1]

    def _is_input_layer(self, layer: Any) -> bool:
        # Avoid importing keras at module import time.
        name = layer.__class__.__name__
        if name == "InputLayer":
            return True
        # Some Functional models may include layers created by keras.Input(...) with no inbound nodes.
        if hasattr(layer, "input") and not hasattr(layer, "output"):
            return True
        return False

    def _convert_to_array_and_meta(
        self, data: Any, model: Any
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], bool]:
        """
        Convert workflow data into a numpy batch array and a parallel metadata list.

        Returns:
            (x, meta_list, is_batch)
        """
        is_batch = isinstance(data, list)
        items: List[Any] = data if isinstance(data, list) else [data]
        if len(items) == 0:
            raise ValueError(tr_("Input data list is empty"))

        meta_list: List[Dict[str, Any]] = [self._extract_meta(item) for item in items]

        # Fast-path for known wrapper types (keeps consistent stacking)
        if isinstance(items[0], AudioData):
            x = np.array([item.data for item in items], dtype=np.float32)  # (batch, channels, samples)
        elif isinstance(items[0], FeatureData):
            x = np.array([item.data for item in items], dtype=np.float32)  # (batch, channels, ...)
        else:
            # Accept numpy arrays / python lists
            x = np.array(items)
            if x.dtype == object:
                raise ValueError(tr_("Input samples have inconsistent shapes; cannot stack into a batch array"))

        x = self._adjust_data_format(x, model)
        return x, meta_list, is_batch

    def _extract_meta(self, item: Any) -> Dict[str, Any]:
        if isinstance(item, AudioData):
            return {
                "sample_rate": int(getattr(item, "sample_rate", 48000) or 48000),
                "hop_length": 512,
                "source_file": getattr(item, "file_path", "") or "",
            }
        if isinstance(item, FeatureData):
            return {
                "sample_rate": int(getattr(item, "sample_rate", 48000) or 48000),
                "hop_length": int(getattr(item, "hop_length", 512) or 512),
                "source_file": getattr(item, "source_file", "") or "",
            }
        return {"sample_rate": 48000, "hop_length": 512, "source_file": ""}

    def _adjust_data_format(self, x: np.ndarray, model: Any) -> np.ndarray:
        """
        If the model is a Keras model, convert input to channels_last when it looks like channels_first.
        """
        if model is None:
            return x

        if not self._is_keras_model(model):
            return x

        # Heuristic conversion similar to Trainer/Predict nodes
        if x.ndim == 3:
            # (batch, channels, length) -> (batch, length, channels)
            if x.shape[1] < x.shape[2]:
                return np.transpose(x, (0, 2, 1))
        elif x.ndim == 4:
            # (batch, channels, height, width) -> (batch, height, width, channels)
            if x.shape[1] < x.shape[2] and x.shape[1] < x.shape[3]:
                return np.transpose(x, (0, 2, 3, 1))
        return x

    def _is_keras_model(self, model: Any) -> bool:
        try:
            from tensorflow import keras  # type: ignore

            return isinstance(model, keras.Model)
        except Exception:
            return False

    def _wrap_outputs_as_feature_data(
        self, y: Any, meta_list: List[Dict[str, Any]], is_batch: bool
    ) -> Union[FeatureData, List[FeatureData]]:
        arr = np.asarray(y)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)

        if arr.ndim < 1:
            raise ValueError(tr_("Unsupported model output shape"))

        # Ensure batch dimension
        if not is_batch:
            arr = arr.reshape((1, *arr.shape))

        if arr.shape[0] != len(meta_list):
            # Fallback: if metadata does not match, repeat first metadata
            if len(meta_list) == 1:
                meta_list = meta_list * int(arr.shape[0])
            else:
                raise ValueError(tr_("Output batch size does not match input batch size"))

        out_list: List[FeatureData] = []
        for i in range(arr.shape[0]):
            out_list.append(self._wrap_single_sample(arr[i], meta_list[i]))

        return out_list if is_batch else out_list[0]

    def _wrap_single_sample(self, sample_out: np.ndarray, meta: Dict[str, Any]) -> FeatureData:
        a = np.asarray(sample_out)
        if a.ndim == 0:
            data = a.reshape(1, 1).astype(np.float32)
        elif a.ndim == 1:
            # (features,) -> (channels=1, features)
            data = a.reshape(1, -1).astype(np.float32)
        elif a.ndim == 2:
            # Heuristic: keep (features, frames) if features <= frames, otherwise transpose.
            feat_frames = a if a.shape[0] <= a.shape[1] else a.T
            data = feat_frames[np.newaxis, :, :].astype(np.float32)  # (1, features, frames)
        else:
            # Flatten all but last dim into frames, last dim is treated as features.
            last = int(a.shape[-1])
            flat = a.reshape(-1, last)  # (frames, features)
            data = flat.T[np.newaxis, :, :].astype(np.float32)  # (1, features, frames)

        return FeatureData(
            data=data,
            feature_type="ai_embedding",
            sample_rate=int(meta.get("sample_rate", 48000)),
            hop_length=int(meta.get("hop_length", 512)),
            source_file=str(meta.get("source_file", "")),
        )

