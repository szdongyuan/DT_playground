"""Focused tests for classification experiment orchestration helpers."""

import json
from io import StringIO

import numpy as np
import pytest
import soundfile as sf

from src.cli.classifier import (
    ClassifierTrainingRequest,
    _encode_labels,
    _prepare_experiment_directory,
    train_classifier,
)
from src.cli.contracts import EventWriter
from src.cli.dataset import load_label_mapping


def test_encode_labels_uses_stable_sorted_class_ids():
    mapping, encoded = _encode_labels({"b.wav": "zebra", "a.wav": "ant"})

    assert mapping == {"ant": 0, "zebra": 1}
    assert encoded == {"a.wav": 0, "b.wav": 1}


def test_load_label_mapping_rejects_duplicate_csv_filenames(tmp_path):
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "filename,label\na.wav,one\na.wav,two\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate filename"):
        load_label_mapping(labels)


def test_load_label_mapping_rejects_duplicate_json_filenames(tmp_path):
    labels = tmp_path / "labels.json"
    labels.write_text('{"a.wav": "one", "a.wav": "two"}', encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate key"):
        load_label_mapping(labels)


def test_experiment_overwrite_requires_compatible_marker(tmp_path):
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    (experiment / "user.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not created by this CLI"):
        _prepare_experiment_directory(experiment, overwrite=True)


def test_experiment_overwrite_accepts_its_own_marker(tmp_path):
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    (experiment / "experiment.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "managed_by": "audio-platform-cli",
                "kind": "classification_experiment",
                "experiment_dir": str(experiment.resolve()),
            }
        ),
        encoding="utf-8",
    )

    _prepare_experiment_directory(experiment, overwrite=True)


def test_train_classifier_records_preflight_input_failure(tmp_path):
    dataset = tmp_path / "audio"
    dataset.mkdir()
    labels = tmp_path / "labels.csv"
    labels.write_text("filename,label\n", encoding="utf-8")
    experiment = tmp_path / "experiment"
    events = StringIO()

    with pytest.raises(ValueError, match="no readable"):
        train_classifier(
            ClassifierTrainingRequest(
                dataset=dataset,
                labels=labels,
                experiment_dir=experiment,
            ),
            writer=EventWriter("jsonl", events),
        )

    marker = json.loads((experiment / "experiment.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in events.getvalue().splitlines()]
    assert marker["status"] == "failed"
    assert records[-1]["event"] == "classification_failed"


def test_train_classifier_completes_tiny_audio_experiment(tmp_path):
    dataset = tmp_path / "audio"
    dataset.mkdir()
    label_lines = ["filename,label"]
    sample_rate = 8000
    samples = np.arange(1024, dtype=np.float32) / sample_rate
    for index in range(16):
        label = "low" if index % 2 == 0 else "high"
        frequency = 220 if label == "low" else 440
        name = f"sample-{index:02d}.wav"
        sf.write(
            dataset / name,
            (0.1 * np.sin(2 * np.pi * frequency * samples)).astype(np.float32),
            sample_rate,
        )
        label_lines.append(f"{name},{label}")
    labels = tmp_path / "labels.csv"
    labels.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
    experiment = tmp_path / "experiment"

    success, payload = train_classifier(
        ClassifierTrainingRequest(
            dataset=dataset,
            labels=labels,
            experiment_dir=experiment,
            sample_rate=sample_rate,
            duration=0.128,
            n_mels=20,
            n_fft=256,
            hop_length=64,
            epochs=1,
            batch_size=4,
            patience=1,
            model_name="tiny",
        ),
        writer=EventWriter("jsonl", StringIO()),
    )

    assert success
    assert payload["status"] == "completed"
    assert payload["class_mapping"] == {"high": 0, "low": 1}
    run_dir = experiment / "run-001"
    manifest_path = run_dir / "manifest.json"
    assert manifest_path.is_file()
    assert (run_dir / "models" / "tiny.keras").is_file()
    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    result = manifest["node_outputs"]["evaluate"]["classification_result"]
    predictions = run_dir / result["artifacts"]["predictions"]
    misclassified = run_dir / result["artifacts"]["misclassified"]
    prediction_rows = [
        json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()
    ]
    assert len(prediction_rows) == result["sample_count"]
    assert all(row["source_ref"].startswith("dataset://") for row in prediction_rows)
    assert misclassified.is_file()
    assert str(tmp_path) not in manifest_text
