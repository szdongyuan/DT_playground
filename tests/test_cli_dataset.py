"""Tests for read-only acoustic dataset inspection."""

import json

import numpy as np
import soundfile as sf

from src.cli.dataset import inspect_dataset


def _write_wave(path, sample_rate=16000, channels=1, duration=0.1):
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = int(sample_rate * duration)
    data = np.zeros((samples, channels), dtype=np.float32)
    if channels == 1:
        data = data[:, 0]
    sf.write(path, data, sample_rate)


def test_inspect_dataset_reports_audio_metadata_and_classes(tmp_path):
    _write_wave(tmp_path / "normal" / "a.wav", sample_rate=16000)
    _write_wave(tmp_path / "abnormal" / "b.wav", sample_rate=8000, channels=2)
    (tmp_path / "broken.wav").write_text("not audio", encoding="utf-8")

    result = inspect_dataset(tmp_path)

    assert result["summary"]["candidate_files"] == 3
    assert result["summary"]["valid_files"] == 2
    assert result["summary"]["invalid_files"] == 1
    assert result["sample_rates"] == {8000: 1, 16000: 1}
    assert result["channels"] == {1: 1, 2: 1}
    assert result["class_distribution"] == {"abnormal": 1, "normal": 1}
    assert "Multiple sample rates were detected." in result["warnings"]


def test_inspect_dataset_checks_external_label_alignment(tmp_path):
    _write_wave(tmp_path / "normal" / "a.wav")
    _write_wave(tmp_path / "normal" / "b.wav")
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"a.wav": "normal", "ghost.wav": "normal"}), encoding="utf-8")

    result = inspect_dataset(tmp_path, labels_path=labels)

    assert result["label_alignment"]["matched_audio_files"] == 1
    assert result["label_alignment"]["missing_labels"] == ["normal/b.wav"]
    assert result["label_alignment"]["orphan_labels"] == ["ghost.wav"]


def test_inspect_empty_dataset_returns_warning(tmp_path):
    result = inspect_dataset(tmp_path)

    assert result["summary"]["valid_files"] == 0
    assert result["warnings"] == ["No supported audio files were found."]
