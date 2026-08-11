# -*- coding: utf-8 -*-
"""Terminal workflow nodes for interactive data visualization."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.ui.i18n import tr_

from ..node_base import BaseNode, NodeCategory, register_node
from ..port import DataType


@dataclass
class CurveSeries:
    """One render-ready line in a multi-curve preview."""

    name: str
    x: np.ndarray
    y: np.ndarray
    source_index: int
    channel_index: Optional[int] = None


@dataclass
class MultiCurvePreviewData:
    """Prepared data and summary statistics for the multi-curve preview."""

    series: List[CurveSeries]
    x_label: str
    interaction_mode: str
    display_mode: str
    aggregate_mode: str
    variability_mode: str
    line_opacity: float
    show_legend: bool
    aggregate_x: Optional[np.ndarray] = None
    aggregate_y: Optional[np.ndarray] = None
    lower_y: Optional[np.ndarray] = None
    upper_y: Optional[np.ndarray] = None
    warnings: List[str] = field(default_factory=list)
    downsampled: bool = False


def _as_numeric_array(value: Any) -> np.ndarray:
    """Convert supported array-like values to a finite real array."""
    array = np.asarray(value)
    if array.dtype.kind not in "biuf":
        raise ValueError(tr_("Only real numeric line data is supported"))
    return array.astype(np.float32, copy=False)


def _clean_line(values: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Return a one-dimensional line with non-finite gaps interpolated."""
    line = _as_numeric_array(values).reshape(-1)
    if line.size == 0:
        raise ValueError(tr_("Empty curves cannot be displayed"))

    finite = np.isfinite(line)
    if not np.any(finite):
        raise ValueError(tr_("A curve contains no finite values"))
    if np.all(finite):
        return line, False

    positions = np.arange(line.size, dtype=np.float32)
    line = np.interp(positions, positions[finite], line[finite]).astype(
        np.float32,
        copy=False,
    )
    return line, True


def _item_name(item: Any, source_index: int) -> str:
    source_path = getattr(item, "file_path", "") or getattr(item, "source_file", "")
    if source_path:
        return os.path.basename(source_path)
    feature_type = getattr(item, "feature_type", "")
    if feature_type:
        return f"{feature_type}_{source_index + 1}"
    return tr_("Series {index}").format(index=source_index + 1)


def _extract_channels(item: Any) -> Tuple[List[np.ndarray], str, str]:
    """Extract channel lines plus their original-axis kind and label."""
    feature_type = getattr(item, "feature_type", "")
    is_feature = bool(feature_type) and hasattr(item, "data")
    is_audio = (
        not is_feature
        and hasattr(item, "data")
        and hasattr(item, "sample_rate")
    )

    raw = getattr(item, "data", item)
    array = _as_numeric_array(raw)

    if is_feature:
        if array.ndim == 1:
            channels = [array]
        elif array.ndim == 2:
            channels = [array[index] for index in range(array.shape[0])]
        elif array.ndim == 3:
            channels = []
            for index in range(array.shape[0]):
                squeezed = np.squeeze(array[index])
                if squeezed.ndim > 1:
                    raise ValueError(
                        tr_(
                            "Feature '{feature_type}' is two-dimensional map data "
                            "and cannot be shown as a curve"
                        ).format(feature_type=feature_type)
                    )
                channels.append(np.atleast_1d(squeezed))
        else:
            raise ValueError(
                tr_(
                    "Feature '{feature_type}' is not line-compatible"
                ).format(feature_type=feature_type)
            )

        if feature_type == "fft":
            return channels, "frequency", tr_("Frequency (Hz)")
        if feature_type == "statistics":
            return channels, "index", tr_("Feature index")
        if int(getattr(item, "hop_length", 0) or 0) > 0:
            return channels, "time", tr_("Time (s)")
        return channels, "index", tr_("Index")

    if is_audio:
        if array.ndim == 1:
            return [array], "time", tr_("Time (s)")
        if array.ndim == 2:
            return [array[index] for index in range(array.shape[0])], "time", tr_("Time (s)")
        raise ValueError(tr_("Audio data must have shape (samples,) or (channels, samples)"))

    if array.ndim == 1:
        return [array], "index", tr_("Index")
    if array.ndim == 2:
        return [array[index] for index in range(array.shape[0])], "index", tr_("Index")
    raise ValueError(tr_("Only one-dimensional lines or channel-by-line arrays are supported"))


def _original_x(item: Any, axis_kind: str, length: int) -> np.ndarray:
    if axis_kind == "frequency":
        sample_rate = float(getattr(item, "sample_rate", 0) or 0)
        return np.linspace(0.0, sample_rate / 2.0, length, dtype=np.float32)
    if axis_kind == "time":
        sample_rate = float(getattr(item, "sample_rate", 0) or 0)
        hop_length = float(getattr(item, "hop_length", 0) or 0)
        step = hop_length / sample_rate if hop_length > 0 and sample_rate > 0 else (
            1.0 / sample_rate if sample_rate > 0 else 1.0
        )
        return np.arange(length, dtype=np.float32) * np.float32(step)
    return np.arange(length, dtype=np.float32)


def _transform_line(line: np.ndarray, mode: str) -> np.ndarray:
    if mode == "mean_center":
        return (line - np.mean(line)).astype(np.float32, copy=False)
    if mode == "minmax":
        span = np.max(line) - np.min(line)
        result = (line - np.min(line)) / span if span > 0 else np.zeros_like(line)
        return result.astype(np.float32, copy=False)
    if mode == "zscore":
        std = np.std(line)
        result = (line - np.mean(line)) / std if std > 0 else np.zeros_like(line)
        return result.astype(np.float32, copy=False)
    return line


def _resample_normalized(
    line: np.ndarray,
    target_x: np.ndarray,
) -> np.ndarray:
    if target_x.size <= 1:
        return np.array([line[0]], dtype=np.float32)
    source_x = np.linspace(0.0, 100.0, line.size, dtype=np.float32)
    return np.interp(target_x, source_x, line).astype(np.float32, copy=False)


def _downsample(x: np.ndarray, y: np.ndarray, max_points: int) -> Tuple[np.ndarray, np.ndarray, bool]:
    if y.size <= max_points:
        return x, y, False
    indices = np.linspace(0, y.size - 1, max_points).astype(np.int64)
    return x[indices], y[indices], True


def _build_aggregate(
    series: Sequence[CurveSeries],
    aggregate_mode: str,
    variability_mode: str,
    percentile_low: float,
    percentile_high: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    if aggregate_mode == "none" or not series:
        return None, None, None
    first_x = series[0].x
    if any(item.x.shape != first_x.shape or not np.allclose(item.x, first_x) for item in series[1:]):
        return None, None, None

    values = np.stack([item.y for item in series], axis=0).astype(
        np.float32,
        copy=False,
    )
    aggregate = (
        np.median(values, axis=0)
        if aggregate_mode == "median"
        else np.mean(values, axis=0)
    )
    aggregate = aggregate.astype(np.float32, copy=False)
    if variability_mode == "std":
        std = np.std(values, axis=0)
        return (
            aggregate,
            (aggregate - std).astype(np.float32, copy=False),
            (aggregate + std).astype(np.float32, copy=False),
        )
    if variability_mode == "percentile":
        return (
            aggregate,
            np.percentile(values, percentile_low, axis=0).astype(np.float32),
            np.percentile(values, percentile_high, axis=0).astype(np.float32),
        )
    if variability_mode == "minmax":
        return (
            aggregate,
            np.min(values, axis=0).astype(np.float32, copy=False),
            np.max(values, axis=0).astype(np.float32, copy=False),
        )
    return aggregate, None, None


def prepare_multi_curve_preview(data: Any, settings: Dict[str, Any]) -> MultiCurvePreviewData:
    """Convert supported input data into render-ready curves and aggregates."""
    if data is None:
        raise ValueError(tr_("No curve data provided"))

    if isinstance(data, (list, tuple)):
        if not data:
            raise ValueError(tr_("No curve data provided"))
        if all(np.isscalar(value) for value in data):
            items = [np.asarray(data)]
        else:
            items = list(data)
    else:
        items = [data]

    channel_mode = settings["channel_mode"]
    selected_channel = int(settings["selected_channel"])
    alignment = settings["x_alignment"]
    transform = settings["value_transform"]
    max_points = int(settings["max_points"])
    warnings: List[str] = []
    raw_series: List[Tuple[str, np.ndarray, np.ndarray, int, Optional[int], str]] = []

    for source_index, item in enumerate(items):
        channels, axis_kind, axis_label = _extract_channels(item)
        if not channels:
            raise ValueError(tr_("A data item contains no channels"))

        if channel_mode == "merge":
            cleaned = [_clean_line(channel)[0] for channel in channels]
            if len({line.size for line in cleaned}) != 1:
                raise ValueError(tr_("Channels must have equal lengths before they can be merged"))
            merged = np.mean(np.stack(cleaned, axis=0), axis=0).astype(
                np.float32,
                copy=False,
            )
            selected = [(merged, None)]
        elif channel_mode == "selected":
            if selected_channel < 0 or selected_channel >= len(channels):
                raise ValueError(
                    tr_("Selected channel {channel} is unavailable for {name}").format(
                        channel=selected_channel,
                        name=_item_name(item, source_index),
                    )
                )
            selected = [(channels[selected_channel], selected_channel)]
        else:
            selected = [(channel, index) for index, channel in enumerate(channels)]

        for values, channel_index in selected:
            line, repaired = _clean_line(values)
            if repaired:
                warnings.append(
                    tr_("Non-finite values were interpolated in {name}").format(
                        name=_item_name(item, source_index)
                    )
                )
            line = _transform_line(line, transform)
            name = _item_name(item, source_index)
            if channel_index is not None and len(channels) > 1:
                name = tr_("{name} · channel {channel}").format(
                    name=name,
                    channel=channel_index + 1,
                )
            x = _original_x(item, axis_kind, line.size)
            raw_series.append((name, x, line, source_index, channel_index, axis_label))

    if not raw_series:
        raise ValueError(tr_("No line-compatible data was found"))

    series: List[CurveSeries] = []
    downsampled = False
    if alignment == "normalized":
        target_count = max(1, min(max_points, max(line.size for _, _, line, _, _, _ in raw_series)))
        shared_x = np.linspace(
            0.0,
            100.0,
            target_count,
            dtype=np.float32,
        )
        for name, _, line, source_index, channel_index, _ in raw_series:
            y = _resample_normalized(line, shared_x)
            downsampled = downsampled or line.size > target_count
            series.append(
                CurveSeries(name, shared_x, y, source_index, channel_index)
            )
        x_label = tr_("Normalized position (%)")
    else:
        axis_labels = {axis_label for *_, axis_label in raw_series}
        x_label = next(iter(axis_labels)) if len(axis_labels) == 1 else tr_("Original coordinate")
        for name, x, line, source_index, channel_index, _ in raw_series:
            x, line, reduced = _downsample(x, line, max_points)
            downsampled = downsampled or reduced
            series.append(CurveSeries(name, x, line, source_index, channel_index))

    aggregate_y, lower_y, upper_y = _build_aggregate(
        series,
        settings["aggregate_mode"],
        settings["variability_mode"],
        float(settings["percentile_low"]),
        float(settings["percentile_high"]),
    )
    aggregate_x = series[0].x if aggregate_y is not None else None
    if settings["aggregate_mode"] != "none" and aggregate_y is None:
        warnings.append(
            tr_("Aggregate statistics require compatible horizontal coordinates")
        )

    return MultiCurvePreviewData(
        series=series,
        x_label=x_label,
        interaction_mode=settings.get("interaction_mode", "performance"),
        display_mode=settings["display_mode"],
        aggregate_mode=settings["aggregate_mode"],
        variability_mode=settings["variability_mode"],
        line_opacity=float(settings["line_opacity"]),
        show_legend=bool(settings["show_legend"]),
        aggregate_x=aggregate_x,
        aggregate_y=aggregate_y,
        lower_y=lower_y,
        upper_y=upper_y,
        warnings=warnings,
        downsampled=downsampled,
    )


@register_node
class MultiCurveViewerNode(BaseNode):
    """Terminal node that prepares all line-compatible input for one chart."""

    node_type = "multi_curve_viewer"
    display_name = tr_("Multi-curve viewer")
    category = NodeCategory.OUTPUT
    subcategory = tr_("Result viewing")
    subcategory_order = 30
    palette_order = 20
    description = tr_("Compare all line-compatible data in one interactive chart")
    icon = "📉"

    def __init__(self, node_id: str = None):
        self._preview_data: Optional[MultiCurvePreviewData] = None
        super().__init__(node_id=node_id)

    def _setup_ports(self):
        self.add_input("data", DataType.ANY, tr_("Line data"))

    def _setup_parameters(self):
        self.add_parameter(
            "interaction_mode", "choice", "performance",
            display_name=tr_("Interaction mode"),
            choices=["performance", "series_toggle"],
        )
        self.add_parameter(
            "display_mode", "choice", "overlay",
            display_name=tr_("Display mode"),
            choices=["overlay", "offset"],
        )
        self.add_parameter(
            "x_alignment", "choice", "normalized",
            display_name=tr_("Horizontal alignment"),
            choices=["normalized", "original"],
        )
        self.add_parameter(
            "channel_mode", "choice", "separate",
            display_name=tr_("Channel handling"),
            choices=["separate", "merge", "selected"],
        )
        self.add_parameter(
            "selected_channel", "int", 0,
            display_name=tr_("Selected channel"),
            min_value=0,
            max_value=255,
        )
        self.add_parameter(
            "value_transform", "choice", "none",
            display_name=tr_("Value transform"),
            choices=["none", "mean_center", "minmax", "zscore"],
        )
        self.add_parameter(
            "aggregate_mode", "choice", "mean",
            display_name=tr_("Aggregate curve"),
            choices=["none", "mean", "median"],
        )
        self.add_parameter(
            "variability_mode", "choice", "std",
            display_name=tr_("Variability band"),
            choices=["none", "std", "percentile", "minmax"],
        )
        self.add_parameter(
            "percentile_low", "float", 25.0,
            display_name=tr_("Lower percentile"),
            min_value=0.0,
            max_value=100.0,
        )
        self.add_parameter(
            "percentile_high", "float", 75.0,
            display_name=tr_("Upper percentile"),
            min_value=0.0,
            max_value=100.0,
        )
        self.add_parameter(
            "max_points", "int", 2000,
            display_name=tr_("Maximum points per curve"),
            min_value=10,
            max_value=200000,
        )
        self.add_parameter(
            "line_opacity", "float", 0.3,
            display_name=tr_("Line opacity"),
            min_value=0.05,
            max_value=1.0,
        )
        self.add_parameter(
            "show_legend", "bool", False,
            display_name=tr_("Show legend"),
        )

    def validate(self) -> Tuple[bool, str]:
        valid, message = super().validate()
        if not valid:
            return valid, message
        if float(self.get_parameter("percentile_low")) >= float(self.get_parameter("percentile_high")):
            return False, tr_("Lower percentile must be smaller than upper percentile")
        return True, ""

    def execute(self) -> bool:
        try:
            settings = dict(self.parameter_values)
            self._preview_data = prepare_multi_curve_preview(
                self.get_input_data("data"),
                settings,
            )
        except (TypeError, ValueError) as error:
            self._preview_data = None
            self.error_message = str(error)
            return False

        self.report_status(
            tr_("Multi-curve viewer ready: {count} curves").format(
                count=len(self._preview_data.series)
            )
        )
        return True

    def get_preview_outputs(self) -> Dict[str, Any]:
        if self._preview_data is None:
            return {}
        return {"curves": self._preview_data}

    def on_reset(self):
        self._preview_data = None


__all__ = [
    "CurveSeries",
    "MultiCurvePreviewData",
    "MultiCurveViewerNode",
    "prepare_multi_curve_preview",
]
