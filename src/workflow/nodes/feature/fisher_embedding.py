"""Load and run FISHER on local, mono waveform inputs."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.ui.i18n import tr_

from .base import AudioData, FeatureData


REVISION = "1fb316962ebb863d570217b12b40fb0ec1e957c5"
MAX_FRAMES = 1024
PATCH_SIZE = 16
MODEL_CONFIG = {
    "model_type": "fisher",
    "band_width": 50,
    "embed_dim": 384,
    "depth": 12,
    "num_heads": 6,
    "max_band_per_sample": 128,
}


def model_files(directory):
    """Check local configuration and required files without loading tensors."""
    directory = Path(directory).expanduser()
    config_path = directory / "config.json"
    weights = directory / "model.safetensors"
    for path in (config_path, weights):
        if not path.is_file():
            raise ValueError(tr_("Required FISHER model file is missing: {path}").format(path=path))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or any(config.get(key) != value for key, value in MODEL_CONFIG.items()):
        raise ValueError(tr_("Expected a FISHER-small-0723 model configuration"))
    return weights


def _waveforms(items, sample_rate, window, hop):
    groups = defaultdict(list)
    for index, item in enumerate(items):
        if not isinstance(item, AudioData) or item.data.ndim != 2 or not item.is_mono:
            raise ValueError(tr_("FISHER requires mono audio. Select one channel before extraction."))
        name = item.file_path or str(index + 1)
        if item.sample_rate != sample_rate:
            raise ValueError(tr_("FISHER expects {rate} Hz audio. Resample this input: {sample}").format(
                rate=sample_rate, sample=name,
            ))
        waveform = np.asarray(item.data[0], dtype=np.float32)
        frames = 1 + (waveform.size - window) // hop
        if frames < PATCH_SIZE or frames > MAX_FRAMES or not np.isfinite(waveform).all():
            raise ValueError(tr_("FISHER audio must contain 16 to 1024 finite STFT frames. Check or segment this input: {sample}").format(
                sample=name,
            ))
        groups[frames].append((index, waveform))
    return groups


def _spectrograms(torch, waveforms, frames, window, hop, hann):
    samples_used = window + (frames - 1) * hop
    centered = []
    for waveform in waveforms:
        waveform = torch.from_numpy(waveform)
        centered.append(waveform[:samples_used] - waveform.mean())
    spectra = torch.stft(
        torch.stack(centered), window, hop_length=hop, window=hann,
        center=False, return_complex=True,
    )
    spectra = (torch.log(spectra.abs() + 1e-10) + 3.017344307886898) / (2.1531635155379805 * 2)
    spectra = spectra.transpose(-2, -1).unsqueeze(1)
    band_width = MODEL_CONFIG["band_width"]
    if spectra.shape[-1] < band_width:
        spectra = torch.nn.functional.pad(spectra, (0, band_width - spectra.shape[-1]))
    return spectra


def extract(input_data, directory, sample_rate, batch_size, report_progress=None):
    """Extract one vector per clip in input order."""
    is_batch = isinstance(input_data, list)
    items = input_data if is_batch else [input_data]
    if not items:
        raise ValueError(tr_("No input data provided"))
    window = 25 * sample_rate // 1000
    hop = 10 * sample_rate // 1000
    groups = _waveforms(items, sample_rate, window, hop)
    weights = model_files(directory)
    try:
        import torch
        from safetensors.torch import load_file
    except (ImportError, OSError) as exc:
        raise RuntimeError(tr_("Install the optional FISHER runtime from requirements-fisher.txt")) from exc

    from src.audio.fisher_model import FisherSmall

    digest = hashlib.sha256()
    with weights.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    extraction = {"adapter": "fisher_small_0723", "reference_revision": REVISION,
                  "weights_sha256": digest.hexdigest(), "sample_rate": sample_rate,
                  "preprocessing": "log_stft_25ms_10ms_v1"}
    model = FisherSmall()
    model.load_state_dict(load_file(str(weights)), strict=True)
    model.eval()
    features = [None] * len(items)
    hann = torch.hann_window(window)
    processed = 0
    progress_step = max(1, len(items) // 20)
    next_report = progress_step
    with torch.inference_mode():
        for frames, group in groups.items():
            for start in range(0, len(group), batch_size):
                indices, waveforms = zip(*group[start:start + batch_size])
                vectors = model(_spectrograms(torch, waveforms, frames, window, hop, hann)).numpy()
                if not np.isfinite(vectors).all():
                    raise ValueError(tr_("FISHER returned non-finite features"))
                for index, vector in zip(indices, vectors):
                    item = items[index]
                    metadata = dict(getattr(item, "metadata", None) or {}, extraction=extraction)
                    features[index] = FeatureData(
                        vector.reshape(1, -1), "fisher_embedding", sample_rate, hop,
                        item.file_path, metadata,
                    )
                processed += len(indices)
                if report_progress and (processed >= next_report or processed == len(items)):
                    report_progress(processed / len(items))
                    next_report = processed + progress_step
    return features if is_batch else features[0]
