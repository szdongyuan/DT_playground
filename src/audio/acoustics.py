# -*- coding: utf-8 -*-
"""Numerical helpers for calibrated engineering sound-level analysis."""

from functools import lru_cache
from typing import List, Tuple

import numpy as np
from scipy import signal

from src.ui.i18n import tr_


DEFAULT_REFERENCE_PRESSURE_PA = 20.0e-6


def calibrate_pressure(
    samples: np.ndarray,
    calibration_factor: float,
    remove_dc: bool = True,
) -> np.ndarray:
    """Convert normalized full-scale samples to pressure in pascals."""
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError(tr_("Audio data must have shape (channels, samples)"))
    if not np.all(np.isfinite(values)):
        raise ValueError(tr_("Audio data contains NaN or infinite values"))
    if not np.isfinite(calibration_factor) or calibration_factor <= 0:
        raise ValueError(tr_("Calibration factor must be a positive finite Pa/FS value"))

    pressure = values * float(calibration_factor)
    if remove_dc:
        pressure = pressure - np.mean(pressure, axis=1, keepdims=True)
    return pressure


def clipping_warnings(
    samples: np.ndarray,
    threshold: float = 0.999999,
    minimum_count: int = 3,
) -> List[str]:
    """Return channel-specific warnings for repeated full-scale samples."""
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    warnings = []
    for channel_index, channel in enumerate(values):
        count = int(np.count_nonzero(np.abs(channel) >= threshold))
        if count >= minimum_count:
            warnings.append(
                tr_("Channel {channel} may be digitally clipped ({count} full-scale samples)").format(
                    channel=channel_index + 1,
                    count=count,
                )
            )
    return warnings


@lru_cache(maxsize=24)
def _weighting_sos(sample_rate: int, weighting: str) -> np.ndarray:
    """Design a normalized digital A- or C-weighting filter."""
    if sample_rate <= 2000:
        raise ValueError(tr_("A/C weighting requires a sample rate above 2000 Hz"))

    frequency_1 = 20.598997
    frequency_2 = 107.65265
    frequency_3 = 737.86223
    frequency_4 = 12194.217
    w1, w2, w3, w4 = [
        2.0 * np.pi * value
        for value in (frequency_1, frequency_2, frequency_3, frequency_4)
    ]

    if weighting == "A":
        zeros = np.zeros(4)
        poles = np.array([-w1, -w1, -w2, -w3, -w4, -w4])
    elif weighting == "C":
        zeros = np.zeros(2)
        poles = np.array([-w1, -w1, -w4, -w4])
    else:
        raise ValueError(tr_("Unsupported frequency weighting: {weighting}").format(
            weighting=weighting
        ))

    digital_zeros, digital_poles, digital_gain = signal.bilinear_zpk(
        zeros,
        poles,
        1.0,
        fs=float(sample_rate),
    )
    sos = signal.zpk2sos(digital_zeros, digital_poles, digital_gain)
    _, response = signal.sosfreqz(sos, worN=[1000.0], fs=float(sample_rate))
    normalization = float(np.abs(response[0]))
    if not np.isfinite(normalization) or normalization <= 0:
        raise ValueError(tr_("Failed to normalize the frequency-weighting filter"))
    sos[0, :3] /= normalization
    return sos


def apply_frequency_weighting(
    pressure: np.ndarray,
    sample_rate: int,
    weighting: str,
) -> np.ndarray:
    """Apply Z, A, or C frequency weighting to pressure channels."""
    if sample_rate <= 0:
        raise ValueError(tr_("Sample rate must be positive"))
    normalized = str(weighting).upper()
    if normalized == "Z":
        return np.asarray(pressure, dtype=np.float64)
    if normalized not in {"A", "C"}:
        raise ValueError(tr_("Frequency weighting must be Z, A, or C"))
    return signal.sosfilt(_weighting_sos(int(sample_rate), normalized), pressure, axis=-1)


def safe_sound_level(
    energy: np.ndarray,
    reference_pressure_pa: float,
    level_floor_db: float,
) -> np.ndarray:
    """Convert pressure-squared energy to dB SPL with a physical level floor."""
    values = np.asarray(energy, dtype=np.float64)
    if not np.isfinite(reference_pressure_pa) or reference_pressure_pa <= 0:
        raise ValueError(tr_("Reference pressure must be positive and finite"))
    if not np.isfinite(level_floor_db):
        raise ValueError(tr_("Sound-level floor must be finite"))
    if not np.all(np.isfinite(values)):
        raise ValueError(tr_("Sound energy contains NaN or infinite values"))

    scale = max(1.0, float(np.max(np.abs(values))) if values.size else 1.0)
    tolerance = np.finfo(np.float64).eps * scale * 32.0
    if np.any(values < -tolerance):
        raise ValueError(tr_("Sound energy cannot be negative"))
    values = np.maximum(values, 0.0)
    reference_energy = float(reference_pressure_pa) ** 2
    minimum_energy = reference_energy * 10.0 ** (float(level_floor_db) / 10.0)
    return 10.0 * np.log10(np.maximum(values, minimum_energy) / reference_energy)


def sliding_leq(
    pressure: np.ndarray,
    sample_rate: int,
    integration_seconds: float,
    step_seconds: float,
    reference_pressure_pa: float,
    level_floor_db: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate centered complete-window sliding Leq curves."""
    if integration_seconds <= 0 or step_seconds <= 0:
        raise ValueError(tr_("Integration time and output step must be positive"))
    window_size = max(1, int(round(integration_seconds * sample_rate)))
    step_size = max(1, int(round(step_seconds * sample_rate)))
    if pressure.shape[1] < window_size:
        raise ValueError(tr_("Audio is shorter than the Leq integration window"))

    squared = np.square(pressure, dtype=np.float64)
    cumulative = np.concatenate(
        [np.zeros((pressure.shape[0], 1), dtype=np.float64), np.cumsum(squared, axis=1)],
        axis=1,
    )
    starts = np.arange(0, pressure.shape[1] - window_size + 1, step_size, dtype=np.int64)
    energy = (cumulative[:, starts + window_size] - cumulative[:, starts]) / window_size
    levels = safe_sound_level(energy, reference_pressure_pa, level_floor_db)
    times = (starts.astype(np.float64) + window_size / 2.0) / float(sample_rate)
    return times, levels


def exponential_sound_level(
    pressure: np.ndarray,
    sample_rate: int,
    time_constant_seconds: float,
    output_step_seconds: float,
    reference_pressure_pa: float,
    level_floor_db: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate a sampled exponential time-weighted sound-level response."""
    if time_constant_seconds <= 0 or output_step_seconds <= 0:
        raise ValueError(tr_("Time constant and output step must be positive"))
    alpha = float(np.exp(-1.0 / (sample_rate * time_constant_seconds)))
    smoothed = signal.lfilter(
        [1.0 - alpha],
        [1.0, -alpha],
        np.square(pressure, dtype=np.float64),
        axis=-1,
    )
    indices = np.arange(
        0,
        pressure.shape[1],
        max(1, int(round(output_step_seconds * sample_rate))),
        dtype=np.int64,
    )
    levels = safe_sound_level(smoothed[:, indices], reference_pressure_pa, level_floor_db)
    return indices.astype(np.float64) / float(sample_rate), levels


def power_spectral_density(
    pressure: np.ndarray,
    sample_rate: int,
    estimator: str,
    window: str,
    fft_size: str,
    welch_segment_length: int,
    welch_overlap_percent: float,
    welch_average: str,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return a one-sided pressure PSD in Pa squared per hertz."""
    sample_count = pressure.shape[1]
    if estimator == "fft":
        nfft = sample_count if fft_size == "auto" else int(fft_size)
        if nfft < sample_count:
            raise ValueError(tr_("FFT size must cover the complete audio record"))
        frequencies, density = signal.periodogram(
            pressure,
            fs=float(sample_rate),
            window=window,
            nfft=nfft,
            detrend=False,
            return_onesided=True,
            scaling="density",
            axis=-1,
        )
        return frequencies, density, sample_count

    segment_length = int(welch_segment_length)
    if segment_length < 2 or segment_length > sample_count:
        raise ValueError(tr_("Welch segment length must be between 2 and the audio length"))
    if not 0.0 <= welch_overlap_percent < 100.0:
        raise ValueError(tr_("Welch overlap must be in the range [0, 100)"))
    overlap = int(round(segment_length * welch_overlap_percent / 100.0))
    overlap = min(overlap, segment_length - 1)
    frequencies, density = signal.welch(
        pressure,
        fs=float(sample_rate),
        window=window,
        nperseg=segment_length,
        noverlap=overlap,
        nfft=segment_length,
        detrend=False,
        return_onesided=True,
        scaling="density",
        average=welch_average,
        axis=-1,
    )
    return frequencies, density, segment_length


def narrowband_levels(
    frequencies: np.ndarray,
    density: np.ndarray,
    minimum_frequency: float,
    maximum_frequency: float,
    reference_pressure_pa: float,
    level_floor_db: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert PSD bins in a requested range to per-resolution-band SPL."""
    if frequencies.size < 2:
        raise ValueError(tr_("The spectrum must contain at least two frequency bins"))
    mask = (frequencies >= minimum_frequency) & (frequencies <= maximum_frequency)
    if not np.any(mask):
        raise ValueError(tr_("The requested frequency range contains no spectrum bins"))
    bin_width = float(frequencies[1] - frequencies[0])
    energy = density[:, mask] * bin_width
    return frequencies[mask], safe_sound_level(energy, reference_pressure_pa, level_floor_db)


def fractional_octave_bands(
    fraction: int,
    minimum_frequency: float,
    maximum_frequency: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exact base-10 center frequencies and band edges."""
    if fraction not in {1, 3, 6, 12}:
        raise ValueError(tr_("Octave-band fraction must be 1, 3, 6, or 12"))
    ratio = 10.0 ** (3.0 / (10.0 * fraction))
    indices = np.arange(-240, 241, dtype=np.int64)
    centers = 1000.0 * ratio ** indices
    edge_factor = np.sqrt(ratio)
    lower = centers / edge_factor
    upper = centers * edge_factor
    tolerance = 0.005
    mask = (
        (centers >= minimum_frequency * (1.0 - tolerance))
        & (centers <= maximum_frequency * (1.0 + tolerance))
    )
    if not np.any(mask):
        raise ValueError(tr_("The requested range contains no fractional-octave bands"))
    return centers[mask], lower[mask], upper[mask]


def fractional_octave_levels(
    frequencies: np.ndarray,
    density: np.ndarray,
    fraction: int,
    minimum_frequency: float,
    maximum_frequency: float,
    reference_pressure_pa: float,
    level_floor_db: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate PSD cells into fractional-octave pressure levels."""
    if frequencies.size < 2:
        raise ValueError(tr_("The spectrum must contain at least two frequency bins"))
    centers, lower_edges, upper_edges = fractional_octave_bands(
        fraction,
        minimum_frequency,
        maximum_frequency,
    )
    bin_width = float(frequencies[1] - frequencies[0])
    complete = upper_edges <= frequencies[-1] + bin_width / 2.0
    centers = centers[complete]
    lower_edges = lower_edges[complete]
    upper_edges = upper_edges[complete]
    if centers.size == 0:
        raise ValueError(tr_("No complete fractional-octave bands fit below Nyquist"))
    narrowest_band = float(np.min(upper_edges - lower_edges))
    if narrowest_band < 2.0 * bin_width:
        sample_rate = 2.0 * float(frequencies[-1])
        required_length = int(np.ceil(2.0 * sample_rate / narrowest_band))
        raise ValueError(
            tr_(
                "Frequency resolution is insufficient for the lowest octave band; "
                "use a Welch segment length of at least {length}"
            ).format(length=required_length)
        )

    midpoints = (frequencies[:-1] + frequencies[1:]) / 2.0
    cell_lower = np.concatenate(([max(0.0, frequencies[0] - bin_width / 2.0)], midpoints))
    cell_upper = np.concatenate((midpoints, [frequencies[-1] + bin_width / 2.0]))
    energies = np.empty((density.shape[0], centers.size), dtype=np.float64)
    for band_index, (lower, upper) in enumerate(zip(lower_edges, upper_edges)):
        overlap = np.maximum(0.0, np.minimum(cell_upper, upper) - np.maximum(cell_lower, lower))
        energies[:, band_index] = np.sum(density * overlap, axis=1)
    return centers, safe_sound_level(energies, reference_pressure_pa, level_floor_db)
