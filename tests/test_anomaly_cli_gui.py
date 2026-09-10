"""Real acoustic workflows through the common GUI engine and CLI entry point."""

import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import joblib
import numpy as np
import pytest
import soundfile as sf

from src.cli.capabilities import get_capabilities
from src.cli.main import main
from src.workflow import Workflow
from src.workflow.anomaly_io import runtime_versions
from src.workflow.engine import WorkflowEngine
from src.workflow.node_base import create_node


def workflow_definition(folder, model_path=None):
    workflow = Workflow("Anomaly integration")
    def add(kind, name, **parameters):
        node = create_node(kind, name)
        for key, value in parameters.items():
            assert node.set_parameter(key, value)[0]
        workflow.add_node(node)
        return node
    add("audio_folder", "audio", folder_path=str(folder), target_sr=8000, auto_label=False)
    add("mfcc", "mfcc", n_fft=256, hop_length=64, n_mels=20, n_mfcc=4)
    add("feature_vectorizer", "vector")
    workflow.connect("audio", "audio", "mfcc", "audio")
    workflow.connect("mfcc", "feature", "vector", "features")
    if model_path is None:
        add("anomaly_detector_trainer", "model", n_estimators=10)
        add("save_anomaly_model", "save")
        workflow.connect("vector", "feature_matrix", "model", "feature_matrix")
        workflow.connect("model", "anomaly_model", "save", "anomaly_model")
        score_node, score_port = "model", "reference_scores"
    else:
        add("load_anomaly_model", "model", model_path=str(model_path), trusted=True)
        add("anomaly_scorer", "score")
        workflow.connect("model", "anomaly_model", "score", "anomaly_model")
        workflow.connect("vector", "feature_matrix", "score", "feature_matrix")
        score_node, score_port = "score", "anomaly_scores"
    add("anomaly_decision", "decision")
    add("export_anomaly_results", "export")
    workflow.connect(score_node, score_port, "decision", "anomaly_scores")
    workflow.connect("decision", "anomaly_result", "export", "anomaly_result")
    return workflow


def test_cli_training_reload_and_gui_engine_match(tmp_path, qapp):
    audio = tmp_path / "audio"
    audio.mkdir()
    rng = np.random.default_rng(42)
    for index in range(101):
        sf.write(audio / f"{index:04d}.wav", rng.normal(0, .05, 800), 8000)
    train = workflow_definition(audio)
    source = tmp_path / "train.json"
    assert train.save(source)
    assert main(["run", str(source), "--run-dir", str(tmp_path / "train")]) == 0
    model_path = "train/anomaly_model.anomaly.zip"
    reload = workflow_definition(audio, model_path)
    source = tmp_path / "reload.json"
    assert reload.save(source)
    assert main(["validate", str(source), "--check-data", "--json"]) == 0
    assert main(["run", str(source), "--run-dir", str(tmp_path / "reload")]) == 0
    train_csv = tmp_path / "train/anomaly_results/results.csv"
    reload_csv = tmp_path / "reload/anomaly_results/results.csv"
    assert train_csv.read_bytes() == reload_csv.read_bytes()
    with reload_csv.open(encoding="utf-8", newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 101
    gui = Workflow.load(source)
    gui.nodes["model"].set_parameter("model_path", str(tmp_path / model_path))
    gui.nodes["export"].set_parameter("output_folder", str(tmp_path / "gui"))
    engine = WorkflowEngine()
    engine.set_workflow(gui)
    result = engine.execute_sync()
    assert result.success, result.message
    assert reload_csv.read_bytes() == (tmp_path / "gui/anomaly_results/results.csv").read_bytes()
    manifest = json.loads((tmp_path / "reload/manifest.json").read_text())
    csv_key = str(Path("anomaly_results") / "results.csv")
    assert csv_key in manifest["artifacts"]["produced_files"]
    assert manifest["artifacts"]["sha256"][csv_key] == hashlib.sha256(reload_csv.read_bytes()).hexdigest()
    assert manifest["node_outputs"]["score"]["anomaly_scores"]["raw_scores"]["shape"] == [101]


def test_gui_properties_and_capabilities_share_defaults(qapp):
    from src.ui.node_editor.property_panel import PropertyPanel

    catalog = {node["type"]: node for node in get_capabilities()["nodes"]}
    panel = PropertyPanel()
    try:
        for kind in ("save_anomaly_model", "load_anomaly_model", "export_anomaly_results"):
            node = create_node(kind)
            panel.set_node(node, node.node_id)
            assert set(panel._widgets) == set(node.parameters)
            assert {p["name"]: p["default_value"] for p in catalog[kind]["parameters"]} == node.parameter_values
            restored = type(node).from_dict(node.to_dict())
            assert restored.parameter_values == node.parameter_values
        node = create_node("load_anomaly_model")
        panel.set_node(node, node.node_id)
        panel.parameter_changed.connect(lambda _, key, value: node.set_parameter(key, value))
        panel._widgets["trusted"].setChecked(True)
        assert node.get_parameter("trusted") is True
        assert type(node).from_dict(node.to_dict()).get_parameter("trusted") is True
        assert create_node("load_model").outputs["model"].data_type.value == "model"
    finally:
        panel.close()


class _ExecutablePayload:
    def __reduce__(self):
        return (eval, ("1 + 1",))


@pytest.mark.parametrize("trusted", [False, True])
def test_cli_check_never_deserializes(tmp_path, monkeypatch, trusted):
    payload = io.BytesIO()
    joblib.dump(_ExecutablePayload(), payload)
    blob = payload.getvalue()
    model = tmp_path / "malicious.anomaly.zip"
    with zipfile.ZipFile(model, "w") as archive:
        archive.writestr("model.joblib", blob)
        archive.writestr("metadata.json", json.dumps({
            "format": "dt-anomaly-model", "format_version": 1,
            "runtime": runtime_versions(), "payload_sha256": hashlib.sha256(blob).hexdigest(),
            "algorithm": "isolation_forest", "feature_count": 1, "reference_count": 2,
            "feature_schema": {}, "training_summary": {}, "score_bounds": [0, 1],
            "score_direction": "higher_is_more_anomalous", "decision_policy": "stored_in_workflow",
        }))
    workflow = Workflow()
    node = create_node("load_anomaly_model")
    node.set_parameter("model_path", str(model))
    node.set_parameter("trusted", trusted)
    workflow.add_node(node)
    source = tmp_path / "workflow.json"
    workflow.save(source)
    monkeypatch.setattr(joblib, "load", lambda *a, **k: pytest.fail("Deserializer was reached"))
    assert main(["validate", str(source), "--check-data", "--json"]) == (0 if trusted else 3)


@pytest.mark.parametrize("parameter,value", [("file_name", "../escape.anomaly.zip"),
                                            ("file_name", "..\\escape.anomaly.zip"),
                                            ("output_folder", "../outside")])
def test_cli_confines_new_model_outputs(tmp_path, parameter, value):
    workflow = workflow_definition(tmp_path)
    workflow.nodes["save"].set_parameter(parameter, value)
    source = tmp_path / "workflow.json"
    workflow.save(source)
    assert main(["run", str(source), "--run-dir", str(tmp_path / "run")]) == 3
    assert not (tmp_path / "outside").exists()


def test_cli_rejects_output_junction_escape(tmp_path):
    import os
    from src.cli.runner import _resolve_workflow_paths
    from src.cli.validation import ValidationReport

    run = tmp_path / "run"
    outside = tmp_path / "outside"
    run.mkdir()
    outside.mkdir()
    link = run / "escape"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        if os.name != "nt":
            pytest.skip(f"Symlink creation unavailable: {exc}")
        import _winapi

        _winapi.CreateJunction(str(outside), str(link))
    workflow = workflow_definition(tmp_path)
    workflow.nodes["save"].set_parameter("output_folder", "escape")
    workflow.nodes["export"].set_parameter("output_folder", "escape")
    report = ValidationReport()
    _resolve_workflow_paths(workflow, tmp_path, run, report)
    assert not report.valid
    assert {e["context"]["node_id"] for e in report.errors} == {"save", "export"}
    assert list(outside.iterdir()) == []
