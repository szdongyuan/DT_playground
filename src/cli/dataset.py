"""Read-only profiling for audio and acoustic datasets."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import soundfile as sf

from src.workflow.nodes.data_source import AudioFolderNode


def inspect_dataset(
    dataset_path: str | Path,
    *,
    recursive: bool = True,
    labels_path: str | Path | None = None,
) -> dict[str, Any]:
    """Profile supported audio files without changing the dataset."""
    root = Path(dataset_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    candidates = root.rglob("*") if recursive else root.glob("*")
    files = sorted(
        path for path in candidates
        if path.is_file() and path.suffix.lower() in AudioFolderNode.SUPPORTED_FORMATS
    )
    valid = []
    invalid = []
    for path in files:
        try:
            info = _read_audio_info(path)
            relative = path.relative_to(root).as_posix()
            valid.append(
                {
                    "path": relative,
                    "format": path.suffix.lower().lstrip("."),
                    "sample_rate": int(info[0]),
                    "channels": int(info[1]),
                    "duration_seconds": float(info[2]),
                    "label": relative.split("/", 1)[0] if "/" in relative else "default",
                }
            )
        except Exception as exc:
            invalid.append(
                {"path": path.relative_to(root).as_posix(), "error": str(exc)}
            )

    external_labels = _load_label_mapping(labels_path) if labels_path else None
    alignment = _label_alignment(valid, external_labels) if external_labels is not None else None
    durations = [item["duration_seconds"] for item in valid]
    warnings = []
    if not files:
        warnings.append("No supported audio files were found.")
    if invalid:
        warnings.append(f"{len(invalid)} audio file(s) could not be read.")
    if len({item["sample_rate"] for item in valid}) > 1:
        warnings.append("Multiple sample rates were detected.")
    if len({item["channels"] for item in valid}) > 1:
        warnings.append("Multiple channel counts were detected.")
    if alignment and (alignment["missing_labels"] or alignment["orphan_labels"]):
        warnings.append("The external label mapping is not fully aligned with the audio files.")

    return {
        "schema_version": "1.0",
        "dataset": str(root),
        "recursive": recursive,
        "supported_extensions": sorted(AudioFolderNode.SUPPORTED_FORMATS),
        "summary": {
            "candidate_files": len(files),
            "valid_files": len(valid),
            "invalid_files": len(invalid),
            "total_duration_seconds": sum(durations),
            "min_duration_seconds": min(durations) if durations else None,
            "max_duration_seconds": max(durations) if durations else None,
            "mean_duration_seconds": mean(durations) if durations else None,
        },
        "sample_rates": dict(sorted(Counter(item["sample_rate"] for item in valid).items())),
        "channels": dict(sorted(Counter(item["channels"] for item in valid).items())),
        "formats": dict(sorted(Counter(item["format"] for item in valid).items())),
        "class_distribution": dict(sorted(Counter(item["label"] for item in valid).items())),
        "label_alignment": alignment,
        "invalid_files": invalid,
        "warnings": warnings,
    }


def _read_audio_info(path: Path) -> tuple[int, int, float]:
    """Read metadata with a decoding fallback for formats unsupported by libsndfile."""
    try:
        info = sf.info(str(path))
        return int(info.samplerate), int(info.channels), float(info.duration)
    except Exception:
        import librosa

        data, sample_rate = librosa.load(str(path), sr=None, mono=False)
        channels = 1 if data.ndim == 1 else int(data.shape[0])
        samples = int(data.shape[-1])
        return int(sample_rate), channels, samples / float(sample_rate)


def _load_label_mapping(path_value: str | Path) -> dict[str, Any]:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Label file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("labels"), dict):
            data = data["labels"]
        if not isinstance(data, dict):
            raise ValueError("JSON labels must be a filename-to-label object.")
        return {str(key).replace("\\", "/"): value for key, value in data.items()}
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "filename" not in reader.fieldnames or "label" not in reader.fieldnames:
                raise ValueError("CSV labels require 'filename' and 'label' columns.")
            return {
                str(row["filename"]).replace("\\", "/"): row["label"]
                for row in reader if row.get("filename")
            }
    raise ValueError("Dataset inspection supports CSV or JSON filename mappings.")


def _label_alignment(files: list[dict[str, Any]], labels: dict[str, Any]) -> dict[str, Any]:
    file_names = {item["path"] for item in files}
    basename_counts = Counter(Path(name).name for name in file_names)
    matched: set[str] = set()
    used_label_keys: set[str] = set()
    matched_labels = []
    ambiguous_basenames = []
    for name in sorted(file_names):
        basename = Path(name).name
        if name in labels:
            key = name
        elif basename in labels and basename_counts[basename] == 1:
            key = basename
        else:
            if basename in labels and basename_counts[basename] > 1:
                ambiguous_basenames.append(basename)
            continue
        matched.add(name)
        used_label_keys.add(key)
        matched_labels.append(str(labels[key]))
    return {
        "label_entries": len(labels),
        "matched_audio_files": len(matched),
        "missing_labels": sorted(file_names - matched),
        "orphan_labels": sorted(set(labels) - used_label_keys),
        "ambiguous_basenames": sorted(set(ambiguous_basenames)),
        "class_distribution": dict(sorted(Counter(matched_labels).items())),
    }
