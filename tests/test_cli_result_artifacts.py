"""Result publication contracts for traceable, bounded CLI runs."""

from __future__ import annotations

import json
from pathlib import Path

from src.cli import runner


class _Node:
    node_type = "audio_folder"

    def __init__(self, root: Path):
        self.root = root

    def get_parameter(self, name):
        return str(self.root) if name == "folder_path" else None


class _Workflow:
    def __init__(self, root: Path):
        self.nodes = {"audio": _Node(root)}


def _classification_result(dataset: Path, count: int = 150):
    predictions = []
    for index in range(count):
        predictions.append(
            {
                "index": index,
                "file_path": str(dataset / "nested" / f"sample-{index}.wav"),
                "true_class_id": index % 2,
                "true_class_name": str(index % 2),
                "predicted_class_id": (index + (index % 3 == 0)) % 2,
                "predicted_class_name": str((index + (index % 3 == 0)) % 2),
                "confidence": 0.75,
            }
        )
    return {
        "schema_version": "1.0",
        "task": "classification",
        "sample_count": count,
        "classes": [{"id": 0, "name": "0"}, {"id": 1, "name": "1"}],
        "metrics": {"accuracy": 2 / 3},
        "confusion_matrix": [[50, 25], [25, 50]],
        "per_class": [],
        "predictions": predictions,
        "warnings": [],
    }


def test_classification_results_are_complete_traceable_and_bounded(tmp_path):
    dataset = tmp_path / "private" / "audio"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    summary = runner._publish_result_artifacts(
        {"evaluate": {"classification_result": _classification_result(dataset)}},
        run_dir,
        _Workflow(dataset),
    )

    published = summary["evaluate"]["classification_result"]
    predictions_path = run_dir / published["artifacts"]["predictions"]
    errors_path = run_dir / published["artifacts"]["misclassified"]
    rows = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines()]
    errors = [json.loads(line) for line in errors_path.read_text(encoding="utf-8").splitlines()]

    assert len(rows) == 150
    assert len(errors) == 50
    assert published["sample_count"] == 150
    assert published["error_count"] == 50
    assert "predictions" not in published
    assert rows[0]["source_ref"] == "dataset://nested/sample-0.wav"
    assert rows[0]["sample_id"]
    assert rows[0]["is_error"] is True
    assert "file_path" not in rows[0]
    assert str(dataset) not in predictions_path.read_text(encoding="utf-8")
    assert len(json.dumps(summary)) < 10_000


def test_run_marker_survives_missing_manifest_and_rejects_unmanaged_directory(tmp_path):
    managed = tmp_path / "managed"
    run_id = runner._prepare_run_directory(managed, overwrite=False)
    (managed / "partial.bin").write_bytes(b"partial")

    assert runner._prepare_run_directory(managed, overwrite=True) == run_id
    assert not (managed / "partial.bin").exists()

    unmanaged = tmp_path / "unmanaged"
    unmanaged.mkdir()
    (unmanaged / "partial.bin").write_bytes(b"partial")
    try:
        runner._prepare_run_directory(unmanaged, overwrite=True)
    except FileExistsError as exc:
        assert "not created by this CLI" in str(exc)
    else:
        raise AssertionError("Unmanaged directory must not be overwritten")


def test_result_artifact_filename_cannot_escape_run_directory(tmp_path):
    dataset = tmp_path / "audio"
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    summary = runner._publish_result_artifacts(
        {"../../outside": {"classification_result": _classification_result(dataset, 1)}},
        run_dir,
        _Workflow(dataset),
    )

    relative = summary["../../outside"]["classification_result"]["artifacts"]["predictions"]
    assert (run_dir / relative).is_file()
    assert (run_dir / relative).resolve().is_relative_to(run_dir.resolve())
    assert not (tmp_path / "outside.predictions.jsonl").exists()


def test_manifest_summary_redacts_paths_in_dictionary_keys(tmp_path):
    dataset = tmp_path / "private" / "audio"
    run_dir = tmp_path / "run"
    value = {str(dataset / "sample.wav"): {"path": str(dataset / "sample.wav")}}

    summary = runner._manifest_safe(value, run_dir, [dataset])

    serialized = json.dumps(summary)
    assert str(tmp_path) not in serialized
    assert "dataset://sample.wav" in serialized
