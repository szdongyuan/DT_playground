"""Persistence, alignment, trust and exclusive publication regressions."""

import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import joblib
import numpy as np
import pytest

from src.workflow import anomaly_io
from src.workflow.node_base import create_node
from src.workflow.nodes.anomaly import AnomalyModelArtifact, AnomalyScoresData, FeatureMatrixData


@pytest.fixture
def fitted():
    features = FeatureMatrixData(np.random.default_rng(42).normal(size=(30, 3)),
                                 [f"sample-{i}" for i in range(30)])
    node = create_node("anomaly_detector_trainer")
    node.set_parameter("n_estimators", 10)
    node.inputs["feature_matrix"].data = features
    assert node.execute(), node.error_message
    return node.outputs["anomaly_model"].data, features


def make_scores(count):
    return AnomalyScoresData(np.linspace(0, 1, count), np.linspace(0, 100, count),
                             [f"domain-{i}/same.wav" for i in range(count)],
                             reference_normalized_scores=np.arange(10.0))


@pytest.mark.parametrize("scaling", ["standard", "robust", "none"])
def test_roundtrip_preserves_scores_and_decisions(tmp_path, fitted, scaling):
    _, features = fitted
    trainer = create_node("anomaly_detector_trainer")
    trainer.set_parameter("n_estimators", 10)
    trainer.set_parameter("scaling", scaling)
    trainer.inputs["feature_matrix"].data = features
    assert trainer.execute()
    original = trainer.outputs["anomaly_model"].data
    save = create_node("save_anomaly_model")
    save.set_parameter("output_folder", str(tmp_path))
    save.inputs["anomaly_model"].data = original
    assert save.execute(), save.error_message
    path = save.outputs["model_path"].data
    load = create_node("load_anomaly_model")
    load.set_parameter("model_path", path)
    assert not load.execute()
    load.set_parameter("trusted", True)
    assert load.execute(), load.error_message
    restored = load.outputs["anomaly_model"].data
    np.testing.assert_array_equal(original.reference_scores, restored.reference_scores)
    np.testing.assert_array_equal(original.transform(features.matrix), restored.transform(features.matrix))
    outputs = []
    for artifact in (original, restored):
        scorer = create_node("anomaly_scorer")
        scorer.inputs["anomaly_model"].data = artifact
        scorer.inputs["feature_matrix"].data = features
        assert scorer.execute()
        scores = scorer.outputs["anomaly_scores"].data
        decision = create_node("anomaly_decision")
        decision.inputs["anomaly_scores"].data = scores
        assert decision.execute()
        outputs.append((scores, decision.outputs["anomaly_result"].data))
    for name in ("raw_scores", "normalized_scores", "reference_normalized_scores"):
        np.testing.assert_array_equal(getattr(outputs[0][0], name), getattr(outputs[1][0], name))
    np.testing.assert_array_equal(outputs[0][1].is_anomaly, outputs[1][1].is_anomaly)
    assert outputs[0][1].threshold == outputs[1][1].threshold
    before = Path(path).read_bytes()
    assert not save.execute()
    assert Path(path).read_bytes() == before


@pytest.mark.parametrize("trust", [False, None, 1, "true"])
def test_trust_refused_before_joblib(tmp_path, monkeypatch, trust):
    def forbidden(*args, **kwargs):
        pytest.fail("Deserializer was reached")
    monkeypatch.setattr(joblib, "load", forbidden)
    with pytest.raises(PermissionError):
        AnomalyModelArtifact.load(str(tmp_path / "missing"), trusted=trust)


def rewrite_bundle(path, metadata_change=None, payload=None):
    with zipfile.ZipFile(path) as archive:
        metadata = json.loads(archive.read("metadata.json"))
        data = archive.read("model.joblib")
    if metadata_change:
        metadata_change(metadata)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("metadata.json", json.dumps(metadata))
        archive.writestr("model.joblib", payload if payload is not None else data)


@pytest.mark.parametrize("change", ["version", "runtime", "hash", "corrupt"])
def test_invalid_bundle_rejected_before_deserialization(tmp_path, fitted, monkeypatch, change):
    model, _ = fitted
    path = tmp_path / "model.zip"
    model.save(path)
    if change == "corrupt":
        path.write_bytes(b"broken")
    else:
        key = {"version": "format_version", "runtime": "runtime", "hash": "payload_sha256"}[change]
        rewrite_bundle(path, lambda m: m.update({key: "invalid"}))
    monkeypatch.setattr(joblib, "load", lambda *a, **k: pytest.fail("Deserializer was reached"))
    with pytest.raises((ValueError, zipfile.BadZipFile)):
        anomaly_io.inspect_model(path)
    with pytest.raises((ValueError, zipfile.BadZipFile)):
        anomaly_io.load_model(path, trusted=True)


def test_legacy_requires_explicit_format_and_never_rewrites(tmp_path, fitted):
    model, _ = fitted
    path = tmp_path / "old.joblib"
    joblib.dump(model, path)
    before = path.read_bytes()
    with pytest.raises(zipfile.BadZipFile):
        anomaly_io.load_model(path, trusted=True)
    with pytest.warns(UserWarning, match="Legacy"):
        restored = anomaly_io.load_model(path, trusted=True, format="legacy_joblib")
    assert restored.feature_count == model.feature_count
    assert path.read_bytes() == before
    joblib.dump({"wrong": "type"}, path)
    with pytest.warns(UserWarning), pytest.raises(TypeError):
        anomaly_io.load_model(path, trusted=True, format="legacy_joblib")


@pytest.mark.parametrize("count", [0, 1, 100, 101, 200])
@pytest.mark.parametrize("decisions", [False, True])
def test_complete_export(tmp_path, count, decisions):
    scores = make_scores(count)
    data = scores
    if decisions and count:
        node = create_node("anomaly_decision")
        node.inputs["anomaly_scores"].data = scores
        assert node.execute()
        data = node.outputs["anomaly_result"].data
    summary = anomaly_io.export_results(data, tmp_path / "result")
    with open(summary["csv_path"], encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == count
    assert [r["sample_id"] for r in rows] == scores.sample_ids
    np.testing.assert_array_equal([float(r["raw_score"]) for r in rows], scores.raw_scores)
    np.testing.assert_array_equal([float(r["normalized_score"]) for r in rows], scores.normalized_scores)
    meta = json.loads(Path(summary["metadata_path"]).read_text(encoding="utf-8"))
    assert meta["sample_count"] == count
    assert meta["csv_sha256"] == anomaly_io.sha256_file(summary["csv_path"])
    if decisions and count:
        assert [int(r["is_anomaly"]) for r in rows] == data.is_anomaly.tolist()
        assert meta["threshold"] == data.threshold


@pytest.mark.parametrize("safe", [False, True])
def test_csv_special_ids_roundtrip(tmp_path, safe):
    scores = make_scores(7)
    scores.sample_ids = ['引号,"\n.wav', '=1+1.wav', '+1.wav', '-1.wav', '@x.wav', "'x.wav", '\t=x.wav']
    summary = anomaly_io.export_results(scores, tmp_path / "result", spreadsheet_safe=safe)
    with open(summary["csv_path"], encoding="utf-8", newline="") as stream:
        values = [row["sample_id"] for row in csv.DictReader(stream)]
    assert [s[1:] if safe else s for s in values] == scores.sample_ids


@pytest.mark.parametrize("mutation", ["nan", "rank", "length", "ids", "source", "range"])
def test_invalid_export_creates_nothing(tmp_path, mutation):
    scores = make_scores(3)
    if mutation == "nan":
        scores.raw_scores[0] = np.nan
    elif mutation == "rank":
        scores.normalized_scores = scores.normalized_scores[:, None]
    elif mutation == "length":
        scores.raw_scores = scores.raw_scores[:2]
    elif mutation == "ids":
        scores.sample_ids[0] = 7
    elif mutation == "source":
        scores.source_items = ["one"]
    else:
        scores.normalized_scores[0] = 101
    with pytest.raises(ValueError):
        anomaly_io.export_results(scores, tmp_path / "result")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("name", ["../bad", "..\\bad", "/root", "C:\\bad", "C:bad", "a/b", "a\\b", "..", "CON", "x.", "x ", "a:ads"])
def test_invalid_basename(tmp_path, name):
    with pytest.raises(ValueError):
        anomaly_io.output_target(str(tmp_path), name)


def test_export_overwrite_and_failure_cleanup(tmp_path, monkeypatch):
    target = tmp_path / "existing"
    target.mkdir()
    marker = target / "marker"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        anomaly_io.export_results(make_scores(3), target)
    def fail(*a, **k):
        raise OSError("injected publication failure")
    monkeypatch.setattr(anomaly_io.os, "rename", fail)
    with pytest.raises(OSError, match="injected"):
        anomaly_io.export_results(make_scores(3), tmp_path / "new")
    assert list(tmp_path.iterdir()) == [target]
    assert marker.read_text() == "keep"


def test_save_publication_failure_cleanup(tmp_path, fitted, monkeypatch):
    def fail(*a, **k):
        raise OSError("injected publication failure")
    monkeypatch.setattr(anomaly_io.os, "link", fail)
    with pytest.raises(OSError, match="injected"):
        fitted[0].save(tmp_path / "new.zip")
    assert list(tmp_path.iterdir()) == []


def test_export_requires_exclusive_inputs():
    node = create_node("export_anomaly_results")
    assert not node.validate()[0]
    node.inputs["anomaly_scores"].set_connected(True)
    assert node.validate()[0]
    node.inputs["anomaly_result"].set_connected(True)
    assert not node.validate()[0]
    node.inputs["anomaly_scores"].data = make_scores(2)
    node.inputs["anomaly_result"].data = make_scores(2)
    assert not node.execute()


@pytest.mark.parametrize("field,value", [("feature_count", 8), ("score_bounds", (1, 0)),
                                         ("reference_scores", [np.nan, 1]),
                                         ("feature_schema", {"feature_count": 99})])
def test_invalid_model_state_cannot_be_saved(tmp_path, fitted, field, value):
    model, _ = fitted
    setattr(model, field, value)
    with pytest.raises(ValueError):
        model.save(tmp_path / "invalid.zip")
    assert list(tmp_path.iterdir()) == []


def test_export_mid_write_failure_removes_partial_results(tmp_path, monkeypatch):
    original = anomaly_io._json_bytes
    calls = 0
    def fail_second(value):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("metadata write failed")
        return original(value)
    monkeypatch.setattr(anomaly_io, "_json_bytes", fail_second)
    with pytest.raises(OSError):
        anomaly_io.export_results(make_scores(200), tmp_path / "result")
    assert list(tmp_path.iterdir()) == []


def test_save_racing_destination_is_preserved(tmp_path, fitted, monkeypatch):
    original = anomaly_io.os.link
    target = tmp_path / "model.zip"
    def racing_link(source, destination):
        target.write_bytes(b"another writer")
        original(source, destination)
    monkeypatch.setattr(anomaly_io.os, "link", racing_link)
    with pytest.raises(FileExistsError):
        fitted[0].save(target)
    assert target.read_bytes() == b"another writer"
    assert list(tmp_path.iterdir()) == [target]


def test_export_existing_symlink_is_not_followed(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    target = tmp_path / "link"
    try:
        target.symlink_to(original, target_is_directory=True)
    except OSError as exc:
        if anomaly_io.os.name != "nt":
            pytest.skip(f"Symlink creation unavailable: {exc}")
        import _winapi

        _winapi.CreateJunction(str(original), str(target))
    with pytest.raises(FileExistsError):
        anomaly_io.export_results(make_scores(2), target)
    assert list(original.iterdir()) == []


def test_manifest_summarizes_without_deepcopy():
    from dataclasses import dataclass
    from src.cli.contracts import json_safe

    class NoCopy:
        def __deepcopy__(self, memo):
            pytest.fail("Large model/audio state must not be deep-copied for a summary")
    @dataclass
    class Wrapped:
        state: object
        array: np.ndarray
    result = json_safe(Wrapped(NoCopy(), np.zeros(200)))
    assert result["state"] == {"type": "NoCopy"}
    assert result["array"]["shape"] == [200]
