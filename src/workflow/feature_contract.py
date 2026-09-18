"""Serializable feature preprocessing and provenance shared by workflow nodes."""

from src.ui.i18n import tr_
from dataclasses import dataclass, replace
import hashlib
import json

import numpy as np


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def temporal_layout(item):
    """Describe feature axes without including file identity or frame count."""
    array = np.asarray(item.data)
    metadata = getattr(item, "metadata", {}) or {}
    return {"shape": list(array.shape[:-1]), "feature_type": item.feature_type,
            "sample_rate": item.sample_rate, "hop_length": item.hop_length,
            "extraction": metadata.get("extraction", {}),
            "preprocessing": metadata.get("preprocessing", "")}


@dataclass
class FeatureStandardizationState:
    """Fitted statistics; applying this state never updates them."""

    layout: dict
    mean: np.ndarray
    scale: np.ndarray
    epsilon: float
    mode: str

    @property
    def fingerprint(self):
        return signature(self.to_dict())

    def to_dict(self):
        return {"version": 1, "layout": self.layout, "mean": self.mean.tolist(),
                "scale": self.scale.tolist(), "epsilon": self.epsilon, "mode": self.mode}

    @classmethod
    def from_dict(cls, value):
        if value.get("version") != 1:
            raise ValueError(tr_("Unsupported standardization state version"))
        result = cls(value["layout"], np.asarray(value["mean"], dtype=float),
                     np.asarray(value["scale"], dtype=float), float(value["epsilon"]), value["mode"])
        if (result.mode not in ("matrix", "temporal") or result.epsilon <= 0
                or not np.isfinite(result.epsilon) or result.mean.shape != result.scale.shape
                or not np.isfinite(result.mean).all() or not np.isfinite(result.scale).all()
                or np.any(result.scale < result.epsilon)):
            raise ValueError(tr_("Invalid standardization state"))
        return result


def standardize(data, state=None, epsilon=1e-6):
    """Fit per-column/per-band statistics, or strictly reuse a fitted state."""
    from .nodes.anomaly import FeatureMatrixData
    from .nodes.feature import FeatureData

    if epsilon <= 0 or not np.isfinite(epsilon):
        raise ValueError(tr_("Standardization epsilon must be positive and finite"))
    matrix_mode = isinstance(data, FeatureMatrixData)
    single_sample = isinstance(data, FeatureData)
    if single_sample:
        data = [data]
    if matrix_mode:
        arrays = [data.matrix]
        layout = {"feature_count": data.matrix.shape[1], "schema": data.schema}
        mode = "matrix"
    else:
        if not isinstance(data, (list, tuple)) or not data:
            raise ValueError(tr_("Temporal standardization requires a non-empty FeatureData list"))
        layout = temporal_layout(data[0])
        arrays = [np.asarray(item.data, dtype=float) for item in data]
        if any(array.ndim < 2 or temporal_layout(item) != layout
               for array, item in zip(arrays, data)):
            raise ValueError(tr_("Temporal feature layouts must match"))
        mode = "temporal"
    if any(not array.size or not np.isfinite(array).all() for array in arrays):
        raise ValueError(tr_("Features must be non-empty and finite"))
    if state is None:
        if (layout.get("preprocessing") or layout.get("schema", {}).get("preprocessing")
                or layout.get("schema", {}).get("layout", {}).get("preprocessing")):
            raise ValueError(tr_("Features are already standardized"))
        if matrix_mode:
            mean, std = arrays[0].mean(axis=0), arrays[0].std(axis=0)
        else:
            # Merge moments without materializing every training frame together.
            count, mean, m2 = 0, np.zeros(arrays[0].shape[:-1]), np.zeros(arrays[0].shape[:-1])
            for array in arrays:
                n = array.shape[-1]
                local_mean = array.mean(axis=-1)
                delta = local_mean - mean
                m2 += array.var(axis=-1) * n + delta ** 2 * count * n / (count + n)
                mean += delta * n / (count + n)
                count += n
            std = np.sqrt(m2 / count)
        state = FeatureStandardizationState(layout, mean, np.maximum(std, epsilon), epsilon, mode)
    else:
        state = FeatureStandardizationState.from_dict(state.to_dict())
        if state.mode != mode or state.layout != layout:
            raise ValueError(tr_("Feature layout differs from fitted standardization state"))
        expected_shape = arrays[0].shape[1:] if matrix_mode else arrays[0].shape[:-1]
        if state.mean.shape != expected_shape:
            raise ValueError(tr_("Invalid standardization state"))
    if matrix_mode:
        schema = dict(data.schema, preprocessing=state.fingerprint)
        result = replace(data, matrix=(data.matrix - state.mean) / state.scale, schema=schema)
    else:
        result = [replace(item, data=(array - state.mean[..., None]) / state.scale[..., None],
                          metadata=dict(item.metadata or {}, preprocessing=state.fingerprint))
                  for item, array in zip(data, arrays)]
    return (result[0] if single_sample else result), state


def aggregate_scores(scores, parent_ids, method, fraction=0.1):
    """Aggregate unscaled scores while preserving first-seen file order."""
    values = np.asarray(scores, dtype=float)
    if len(values) != len(parent_ids) or not np.isfinite(values).all():
        raise ValueError(tr_("Scores and parent IDs must be finite and aligned"))
    if method not in ("mean", "max", "top_fraction_mean") or not 0 < fraction <= 1:
        raise ValueError(tr_("Invalid score aggregation settings"))
    groups = {}
    for index, parent in enumerate(parent_ids):
        groups.setdefault(parent, []).append(index)
    result = []
    for indices in groups.values():
        group = values[indices]
        if method == "top_fraction_mean":
            group = np.sort(group)[-max(1, int(np.ceil(len(group) * fraction))):]
        result.append(float(group.max() if method == "max" else group.mean()))
    return np.asarray(result), list(groups), [indices[0] for indices in groups.values()]
