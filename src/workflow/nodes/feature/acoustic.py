# -*- coding: utf-8 -*-
"""Calibrated sound-pressure-level analysis and curve conversion nodes."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np

from src.audio.acoustics import (
    apply_frequency_weighting,
    calibrate_pressure,
    clipping_warnings,
    exponential_sound_level,
    fractional_octave_levels,
    narrowband_levels,
    power_spectral_density,
    sliding_leq,
)
from src.ui.i18n import tr_
from src.workflow.curve import CurveData

from ...node_base import BaseNode, NodeCategory, register_node
from ...port import DataType
from ..data_source import AudioData
from .base import FeatureData, validate_audio_input


def _as_audio_items(value: Union[AudioData, List[AudioData]]) -> Tuple[List[AudioData], bool]:
    """Return audio records and whether the caller supplied a batch."""
    return (list(value), True) if isinstance(value, list) else ([value], False)


def _as_curve_items(value: Any) -> Tuple[List[CurveData], bool]:
    """Validate curve input and preserve single-versus-batch semantics."""
    is_batch = isinstance(value, (list, tuple))
    items = list(value) if is_batch else [value]
    if not items:
        raise ValueError(tr_("No curve data provided"))
    if not all(isinstance(item, CurveData) for item in items):
        raise TypeError(tr_("Curve input must contain CurveData objects"))
    return items, is_batch


class _SoundLevelNode(BaseNode):
    """Shared configuration and pressure preparation for SPL nodes."""

    def _setup_common_parameters(self):
        self.add_parameter(
            "calibration_factor",
            "float",
            None,
            display_name=tr_("Calibration factor (Pa/FS)"),
            description=tr_("Required pressure represented by digital full scale"),
            min_value=1.0e-30,
        )
        self.add_parameter(
            "reference_pressure_upa",
            "float",
            20.0,
            display_name=tr_("Reference pressure (μPa)"),
            min_value=1.0e-12,
            max_value=1.0e12,
        )
        self.add_parameter(
            "frequency_weighting",
            "choice",
            "Z",
            display_name=tr_("Frequency weighting"),
            choices=["Z", "A", "C"],
        )
        self.add_parameter(
            "remove_dc",
            "bool",
            True,
            display_name=tr_("Remove DC"),
        )
        self.add_parameter(
            "level_floor_db",
            "float",
            -200.0,
            display_name=tr_("Sound-level floor (dB)"),
            min_value=-400.0,
            max_value=100.0,
        )
        self.add_parameter(
            "detect_clipping",
            "bool",
            True,
            display_name=tr_("Detect clipping"),
        )

    def _common_values(self) -> Tuple[float, float, str, float]:
        calibration = self.get_parameter("calibration_factor")
        if calibration is None or not np.isfinite(calibration) or calibration <= 0:
            raise ValueError(tr_("Calibration factor must be entered as a positive Pa/FS value"))
        reference_pressure = float(self.get_parameter("reference_pressure_upa")) * 1.0e-6
        weighting = str(self.get_parameter("frequency_weighting")).upper()
        floor = float(self.get_parameter("level_floor_db"))
        return float(calibration), reference_pressure, weighting, floor

    def _prepare_audio(self, audio: AudioData) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        calibration, reference_pressure, weighting, floor = self._common_values()
        if int(audio.sample_rate) <= 0:
            raise ValueError(tr_("Sample rate must be positive"))
        warnings = (
            clipping_warnings(audio.data)
            if bool(self.get_parameter("detect_clipping"))
            else []
        )
        pressure = calibrate_pressure(
            audio.data,
            calibration,
            bool(self.get_parameter("remove_dc")),
        )
        weighted = apply_frequency_weighting(pressure, int(audio.sample_rate), weighting)
        metadata = {
            "sample_rate": int(audio.sample_rate),
            "calibration_factor_pa_per_fs": calibration,
            "reference_pressure_pa": reference_pressure,
            "frequency_weighting": weighting,
            "remove_dc": bool(self.get_parameter("remove_dc")),
            "level_floor_db": floor,
            "warnings": list(warnings),
        }
        return weighted, warnings, metadata

    @staticmethod
    def _channel_names(audio: AudioData) -> List[str]:
        return [f"Ch {index + 1}" for index in range(audio.channels)]


@register_node
class TimeVaryingSoundLevelNode(_SoundLevelNode):
    """Calculate per-channel sliding Leq or exponential sound-level curves."""

    node_type = "time_varying_sound_level"
    display_name = tr_("Time-varying total sound level")
    category = NodeCategory.FEATURE
    subcategory = tr_("Acoustic analysis")
    subcategory_order = 5
    palette_order = 10
    description = tr_("Calculate calibrated time-varying total sound pressure level")
    icon = "🔊"

    def _setup_ports(self):
        self.add_input("audio", DataType.AUDIO, tr_("Audio"))
        self.add_output("curve", DataType.CURVE, tr_("Sound-level curve"))

    def _setup_parameters(self):
        self._setup_common_parameters()
        self.add_parameter(
            "sound_level_correction_db",
            "float",
            0.0,
            display_name=tr_("Sound-level correction (dB)"),
            min_value=-200.0,
            max_value=200.0,
        )
        self.add_parameter(
            "calculation_mode",
            "choice",
            "sliding_leq",
            display_name=tr_("Calculation mode"),
            choices=["sliding_leq", "exponential"],
        )
        self.add_parameter(
            "integration_time_seconds",
            "float",
            1.0,
            display_name=tr_("Integration time (s)"),
            min_value=0.001,
            max_value=3600.0,
            visible_when={"calculation_mode": "sliding_leq"},
        )
        self.add_parameter(
            "leq_step_seconds",
            "float",
            0.1,
            display_name=tr_("Output step (s)"),
            min_value=0.0001,
            max_value=3600.0,
            visible_when={"calculation_mode": "sliding_leq"},
        )
        self.add_parameter(
            "time_weighting",
            "choice",
            "fast",
            display_name=tr_("Time weighting"),
            choices=["fast", "slow"],
            visible_when={"calculation_mode": "exponential"},
        )
        self.add_parameter(
            "exponential_step_seconds",
            "float",
            0.05,
            display_name=tr_("Output step (s)"),
            min_value=0.0001,
            max_value=3600.0,
            visible_when={"calculation_mode": "exponential"},
        )

    def execute(self) -> bool:
        try:
            audio_input = validate_audio_input(self.get_input_data("audio"), self.display_name)
            audio_items, is_batch = _as_audio_items(audio_input)
            curves = [self._calculate_item(audio) for audio in audio_items]
        except (TypeError, ValueError) as error:
            self.error_message = str(error)
            return False

        self.set_output_data("curve", curves if is_batch else curves[0])
        warning_count = sum(len(curve.metadata.get("warnings", [])) for curve in curves)
        self.report_status(
            tr_("Time-varying sound level ready: {count} record(s), {warnings} warning(s)").format(
                count=len(curves),
                warnings=warning_count,
            )
        )
        return True

    def _calculate_item(self, audio: AudioData) -> CurveData:
        pressure, warnings, metadata = self._prepare_audio(audio)
        _, reference_pressure, _, floor = self._common_values()
        correction = float(self.get_parameter("sound_level_correction_db"))
        if not np.isfinite(correction):
            raise ValueError(tr_("Sound-level correction must be finite"))
        mode = self.get_parameter("calculation_mode")
        if mode == "sliding_leq":
            integration = float(self.get_parameter("integration_time_seconds"))
            step = float(self.get_parameter("leq_step_seconds"))
            x, levels = sliding_leq(
                pressure,
                int(audio.sample_rate),
                integration,
                step,
                reference_pressure,
                floor,
            )
            curve_type = "time_sliding_leq"
            metadata.update({
                "calculation_mode": mode,
                "integration_time_seconds": integration,
                "output_step_seconds": step,
                "time_coordinate": "window_center",
            })
        else:
            time_weighting = self.get_parameter("time_weighting")
            time_constant = 0.125 if time_weighting == "fast" else 1.0
            step = float(self.get_parameter("exponential_step_seconds"))
            x, levels = exponential_sound_level(
                pressure,
                int(audio.sample_rate),
                time_constant,
                step,
                reference_pressure,
                floor,
            )
            curve_type = f"time_{time_weighting}"
            metadata.update({
                "calculation_mode": mode,
                "time_weighting": time_weighting,
                "time_constant_seconds": time_constant,
                "output_step_seconds": step,
                "startup_transient_seconds": time_constant,
            })
        levels = np.maximum(levels + correction, floor)
        metadata["sound_level_correction_db"] = correction
        metadata["warnings"] = warnings
        return CurveData(
            data=levels,
            x=x,
            x_name=tr_("Time"),
            x_unit="s",
            y_name=tr_("Sound pressure level"),
            y_unit="dB SPL",
            curve_type=curve_type,
            channel_names=self._channel_names(audio),
            source_file=audio.file_path,
            metadata=metadata,
        )


@register_node
class SteadyStateFrequencySoundLevelNode(_SoundLevelNode):
    """Calculate per-channel narrowband or fractional-octave SPL curves."""

    node_type = "steady_state_frequency_sound_level"
    display_name = tr_("Steady-state frequency sound level")
    category = NodeCategory.FEATURE
    subcategory = tr_("Acoustic analysis")
    subcategory_order = 5
    palette_order = 20
    description = tr_("Calculate calibrated narrowband or fractional-octave sound pressure levels")
    icon = "📐"

    def _setup_ports(self):
        self.add_input("audio", DataType.AUDIO, tr_("Audio"))
        self.add_output("curve", DataType.CURVE, tr_("Sound-level curve"))

    def _setup_parameters(self):
        self._setup_common_parameters()
        self.add_parameter(
            "analysis_type",
            "choice",
            "octave",
            display_name=tr_("Analysis type"),
            choices=["narrowband", "octave"],
        )
        self.add_parameter(
            "spectral_estimator",
            "choice",
            "welch",
            display_name=tr_("Spectral estimator"),
            choices=["fft", "welch"],
            visible_when={"analysis_type": "narrowband"},
        )
        self.add_parameter(
            "fft_size",
            "choice",
            "auto",
            display_name=tr_("FFT size"),
            choices=["auto", "512", "1024", "2048", "4096", "8192", "16384", "32768", "65536", "131072", "262144"],
            visible_when={"analysis_type": "narrowband", "spectral_estimator": "fft"},
        )
        self.add_parameter(
            "window",
            "choice",
            "hann",
            display_name=tr_("Window"),
            choices=["hann", "hamming", "blackman", "boxcar"],
        )
        welch_visibility = [
            {"analysis_type": "octave"},
            {"analysis_type": "narrowband", "spectral_estimator": "welch"},
        ]
        self.add_parameter(
            "welch_segment_length",
            "int",
            4096,
            display_name=tr_("Welch segment length"),
            min_value=64,
            max_value=1048576,
            visible_when=welch_visibility,
        )
        self.add_parameter(
            "welch_overlap_percent",
            "float",
            50.0,
            display_name=tr_("Welch overlap (%)"),
            min_value=0.0,
            max_value=99.9,
            visible_when=welch_visibility,
        )
        self.add_parameter(
            "welch_average",
            "choice",
            "mean",
            display_name=tr_("Welch average"),
            choices=["mean", "median"],
            visible_when=welch_visibility,
        )
        self.add_parameter(
            "octave_fraction",
            "choice",
            "1/3",
            display_name=tr_("Octave-band fraction"),
            choices=["1/1", "1/3", "1/6", "1/12"],
            visible_when={"analysis_type": "octave"},
        )
        self.add_parameter(
            "minimum_frequency_hz",
            "float",
            20.0,
            display_name=tr_("Minimum frequency (Hz)"),
            min_value=0.001,
            max_value=1000000.0,
        )
        self.add_parameter(
            "maximum_frequency_hz",
            "float",
            20000.0,
            display_name=tr_("Maximum frequency (Hz)"),
            min_value=0.001,
            max_value=1000000.0,
        )

    def validate(self):
        valid, message = super().validate()
        if not valid:
            return valid, message
        if float(self.get_parameter("minimum_frequency_hz")) >= float(
            self.get_parameter("maximum_frequency_hz")
        ):
            return False, tr_("Minimum frequency must be lower than maximum frequency")
        return True, ""

    def execute(self) -> bool:
        try:
            audio_input = validate_audio_input(self.get_input_data("audio"), self.display_name)
            audio_items, is_batch = _as_audio_items(audio_input)
            curves = [self._calculate_item(audio) for audio in audio_items]
        except (TypeError, ValueError) as error:
            self.error_message = str(error)
            return False

        self.set_output_data("curve", curves if is_batch else curves[0])
        warning_count = sum(len(curve.metadata.get("warnings", [])) for curve in curves)
        self.report_status(
            tr_("Frequency sound level ready: {count} record(s), {warnings} warning(s)").format(
                count=len(curves),
                warnings=warning_count,
            )
        )
        return True

    def _calculate_item(self, audio: AudioData) -> CurveData:
        pressure, warnings, metadata = self._prepare_audio(audio)
        _, reference_pressure, _, floor = self._common_values()
        minimum = float(self.get_parameter("minimum_frequency_hz"))
        requested_maximum = float(self.get_parameter("maximum_frequency_hz"))
        maximum = min(requested_maximum, float(audio.sample_rate) / 2.0)
        if minimum >= maximum:
            raise ValueError(tr_("Frequency range must overlap the audio Nyquist range"))
        if requested_maximum > maximum:
            warnings.append(
                tr_("Maximum frequency was limited to the Nyquist frequency ({frequency:.3f} Hz)").format(
                    frequency=maximum
                )
            )

        analysis_type = self.get_parameter("analysis_type")
        estimator = (
            self.get_parameter("spectral_estimator")
            if analysis_type == "narrowband"
            else "welch"
        )
        frequencies, density, effective_length = power_spectral_density(
            pressure,
            int(audio.sample_rate),
            estimator,
            self.get_parameter("window"),
            self.get_parameter("fft_size"),
            int(self.get_parameter("welch_segment_length")),
            float(self.get_parameter("welch_overlap_percent")),
            self.get_parameter("welch_average"),
        )
        metadata.update({
            "analysis_type": analysis_type,
            "spectral_estimator": estimator,
            "window": self.get_parameter("window"),
            "spectral_length": effective_length,
            "minimum_frequency_hz": minimum,
            "maximum_frequency_hz": maximum,
        })

        if analysis_type == "narrowband":
            x, levels = narrowband_levels(
                frequencies,
                density,
                minimum,
                maximum,
                reference_pressure,
                floor,
            )
            curve_type = f"frequency_narrowband_{estimator}"
        else:
            fraction_text = self.get_parameter("octave_fraction")
            fraction = int(str(fraction_text).split("/")[-1])
            x, levels = fractional_octave_levels(
                frequencies,
                density,
                fraction,
                minimum,
                maximum,
                reference_pressure,
                floor,
            )
            curve_type = f"frequency_octave_{fraction}"
            metadata["octave_fraction"] = fraction_text
        metadata["warnings"] = warnings
        return CurveData(
            data=levels,
            x=x,
            x_name=tr_("Frequency"),
            x_unit="Hz",
            y_name=tr_("Sound pressure level"),
            y_unit="dB SPL",
            curve_type=curve_type,
            channel_names=self._channel_names(audio),
            source_file=audio.file_path,
            metadata=metadata,
        )


@register_node
class CurveToFeatureNode(BaseNode):
    """Convert complete physical curves to aligned one-dimensional features."""

    node_type = "curve_to_feature"
    display_name = tr_("Curve to 1D feature")
    category = NodeCategory.FEATURE
    subcategory = tr_("Feature post-processing")
    subcategory_order = 40
    palette_order = 20
    description = tr_("Preserve complete aligned curves as AI-ready one-dimensional features")
    icon = "↔️"

    _STRICT_METADATA_KEYS = (
        "sample_rate",
        "calibration_factor_pa_per_fs",
        "reference_pressure_pa",
        "frequency_weighting",
        "calculation_mode",
        "time_weighting",
        "integration_time_seconds",
        "analysis_type",
        "spectral_estimator",
        "octave_fraction",
    )

    def _setup_ports(self):
        self.add_input("curve", DataType.CURVE, tr_("Curve"))
        self.add_output("feature", DataType.FEATURE_1D, tr_("1D feature"))

    def _setup_parameters(self):
        self.add_parameter(
            "alignment_mode",
            "choice",
            "strict",
            display_name=tr_("Alignment mode"),
            choices=["strict", "fixed_grid"],
        )
        self.add_parameter(
            "target_points",
            "int",
            256,
            display_name=tr_("Target points"),
            min_value=2,
            max_value=1000000,
            visible_when={"alignment_mode": "fixed_grid"},
        )
        self.add_parameter(
            "grid_spacing",
            "choice",
            "linear",
            display_name=tr_("Grid spacing"),
            choices=["linear", "log"],
            visible_when={"alignment_mode": "fixed_grid"},
        )
        self.add_parameter(
            "grid_start",
            "float",
            None,
            display_name=tr_("Grid start (automatic when empty)"),
            required=False,
            visible_when={"alignment_mode": "fixed_grid"},
        )
        self.add_parameter(
            "grid_end",
            "float",
            None,
            display_name=tr_("Grid end (automatic when empty)"),
            required=False,
            visible_when={"alignment_mode": "fixed_grid"},
        )

    def execute(self) -> bool:
        try:
            curves, is_batch = _as_curve_items(self.get_input_data("curve"))
            self._validate_semantics(curves)
            converted = (
                self._strict_convert(curves)
                if self.get_parameter("alignment_mode") == "strict"
                else self._resample_convert(curves)
            )
        except (TypeError, ValueError) as error:
            self.error_message = str(error)
            return False

        self.set_output_data("feature", converted if is_batch else converted[0])
        self.report_status(
            tr_("Curve features ready: {count} sample(s), {points} points").format(
                count=len(converted),
                points=converted[0].data.shape[-1],
            )
        )
        return True

    def _validate_semantics(self, curves: Sequence[CurveData]):
        first = curves[0]
        first_signature = self._signature(first)
        for index, curve in enumerate(curves):
            if not np.all(np.isfinite(curve.data)):
                raise ValueError(tr_("Curve sample {index} contains NaN or infinite values").format(
                    index=index + 1
                ))
            if curve.channels != first.channels:
                raise ValueError(tr_("Curve sample {index} has a different channel count").format(
                    index=index + 1
                ))
            signature = self._signature(curve)
            for field, expected in first_signature.items():
                if signature[field] != expected:
                    raise ValueError(
                        tr_("Curve sample {index} differs in '{field}'").format(
                            index=index + 1,
                            field=field,
                        )
                    )

    def _signature(self, curve: CurveData) -> Dict[str, Any]:
        signature = {
            "curve_type": curve.curve_type,
            "x_unit": curve.x_unit,
            "y_unit": curve.y_unit,
        }
        signature.update({key: curve.metadata.get(key) for key in self._STRICT_METADATA_KEYS})
        return signature

    def _strict_convert(self, curves: Sequence[CurveData]) -> List[FeatureData]:
        reference_x = curves[0].x
        for index, curve in enumerate(curves[1:], start=2):
            if curve.x.shape != reference_x.shape or not np.allclose(
                curve.x,
                reference_x,
                rtol=1.0e-9,
                atol=1.0e-12,
            ):
                raise ValueError(tr_("Curve sample {index} has different physical coordinates").format(
                    index=index
                ))
        return [self._to_feature(curve, curve.data, curve.x, "strict") for curve in curves]

    def _resample_convert(self, curves: Sequence[CurveData]) -> List[FeatureData]:
        if curves[0].curve_type.startswith("frequency_octave_"):
            raise ValueError(tr_("Fractional-octave curves cannot be resampled between band grids"))
        start_value = self.get_parameter("grid_start")
        end_value = self.get_parameter("grid_end")
        start = float(start_value) if start_value is not None else max(float(curve.x[0]) for curve in curves)
        end = float(end_value) if end_value is not None else min(float(curve.x[-1]) for curve in curves)
        if not np.isfinite(start) or not np.isfinite(end) or start >= end:
            raise ValueError(tr_("The fixed curve grid must have a finite increasing range"))
        if any(start < curve.x[0] or end > curve.x[-1] for curve in curves):
            raise ValueError(tr_("The fixed curve grid must lie inside every source curve"))
        points = int(self.get_parameter("target_points"))
        spacing = self.get_parameter("grid_spacing")
        if spacing == "log":
            if start <= 0:
                raise ValueError(tr_("Logarithmic curve grids require a positive start"))
            target_x = np.geomspace(start, end, points, dtype=np.float64)
        else:
            target_x = np.linspace(start, end, points, dtype=np.float64)
        converted = []
        for curve in curves:
            values = np.stack(
                [np.interp(target_x, curve.x, channel) for channel in curve.data],
                axis=0,
            )
            converted.append(self._to_feature(curve, values, target_x, "fixed_grid"))
        return converted

    @staticmethod
    def _to_feature(
        curve: CurveData,
        values: np.ndarray,
        target_x: np.ndarray,
        alignment_mode: str,
    ) -> FeatureData:
        return FeatureData(
            data=np.asarray(values, dtype=np.float32),
            feature_type=f"curve_{curve.curve_type}",
            sample_rate=int(curve.metadata.get("sample_rate", 0) or 0),
            hop_length=0,
            source_file=curve.source_file,
            metadata={
                "source_curve_type": curve.curve_type,
                "x": np.asarray(target_x, dtype=np.float64).copy(),
                "x_name": curve.x_name,
                "x_unit": curve.x_unit,
                "y_name": curve.y_name,
                "y_unit": curve.y_unit,
                "alignment_mode": alignment_mode,
                "curve_metadata": dict(curve.metadata),
            },
        )


__all__ = [
    "CurveToFeatureNode",
    "SteadyStateFrequencySoundLevelNode",
    "TimeVaryingSoundLevelNode",
]
