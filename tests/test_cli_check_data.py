"""Real-file preflight regressions without workflow execution or artifacts."""

import json

import numpy as np
import pytest
import soundfile as sf

from src.cli.main import main
from src.cli.validation import validate_workflow_file


def write_workflow(tmp_path, parameters=None, labels=None, alignment=None):
    audio = tmp_path / "audio"
    audio.mkdir(exist_ok=True)
    nodes = [{"id": "source", "type": "audio_folder", "parameters": {"folder_path": "audio", **(parameters or {})}}]
    connections = []
    if labels is not None:
        (tmp_path / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
        nodes.extend([
            {"id": "labels", "type": "label_file", "parameters": {"file_path": "labels.json"}},
            {"id": "align", "type": "align_targets", "parameters": alignment or {}},
        ])
        connections = [
            {"source": {"node": "source", "port": "file_paths"}, "target": {"node": "align", "port": "file_paths"}},
            {"source": {"node": "labels", "port": "label_map"}, "target": {"node": "align", "port": "target_map"}},
        ]
    path = tmp_path / "workflow.json"
    path.write_text(json.dumps({"nodes": nodes, "connections": connections}), encoding="utf-8")
    return path


def wave(path, samples=800, channels=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros((samples, channels), dtype=np.float32), 8000)


def codes(report):
    return {item["code"] for item in report.errors}


def test_default_validation_does_not_decode(tmp_path, monkeypatch):
    path = write_workflow(tmp_path)
    (tmp_path / "audio/broken.wav").write_bytes(b"broken")

    def forbidden(*args, **kwargs):
        pytest.fail("Default validation must not decode audio")

    monkeypatch.setattr("src.cli.data_validation._decode_info", forbidden)
    report, _ = validate_workflow_file(path)
    assert report.valid
    assert "data_checks" not in report.to_dict()


def test_empty_folder_is_explicit_validation_failure(tmp_path, capsys):
    path = write_workflow(tmp_path)
    assert main(["validate", str(path), "--json"]) == 0
    capsys.readouterr()
    assert main(["validate", str(path), "--check-data", "--json"]) == 3
    result = json.loads(capsys.readouterr().err)
    assert result["workflow"]["data_checks"]["sources"][0]["valid_files"] == 0
    assert not result["valid"]


@pytest.mark.parametrize("kind", ["corrupt", "empty", "nonfinite"])
def test_bad_audio_is_rejected(tmp_path, kind):
    path = write_workflow(tmp_path)
    audio = tmp_path / "audio/bad.wav"
    if kind == "corrupt":
        audio.write_bytes(b"not a wave")
    elif kind == "empty":
        wave(audio, samples=0)
    else:
        sf.write(audio, np.array([np.nan], dtype=np.float32), 8000, subtype="FLOAT")
    report, _ = validate_workflow_file(path, check_data=True)
    assert {"data.unreadable_audio", "data.empty_audio"} <= codes(report)


def test_mixed_good_and_corrupt_audio_fails(tmp_path):
    path = write_workflow(tmp_path)
    wave(tmp_path / "audio/good.wav")
    (tmp_path / "audio/bad.wav").write_bytes(b"broken")
    report, _ = validate_workflow_file(path, check_data=True)
    assert "data.unreadable_audio" in codes(report)
    assert report.data_checks["sources"][0]["valid_files"] == 1


def test_readonly_alignment_success(tmp_path, monkeypatch):
    path = write_workflow(tmp_path, labels={"version": "1.0", "labels": {"a.wav": 0}})
    wave(tmp_path / "audio/a.wav")
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}

    def forbidden(*args, **kwargs):
        pytest.fail("Preflight must not execute the workflow or load audio nodes")

    monkeypatch.setattr("src.cli.runner.execute_workflow", forbidden)
    monkeypatch.setattr("src.workflow.nodes.data_source.AudioFolderNode.execute", forbidden)
    report, loaded = validate_workflow_file(path, check_data=True)
    assert report.valid
    assert report.data_checks["alignments"][0]["matched_count"] == 1
    assert report.data_checks["sources"][0]["raw_shapes"] == [{"shape": [1, 800], "count": 1}]
    assert loaded.nodes["source"].get_parameter("folder_path") == "audio"
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


@pytest.mark.parametrize("policy,valid", [("error", False), ("drop", True), ("fill", True)])
def test_alignment_respects_missing_policy(tmp_path, policy, valid):
    path = write_workflow(tmp_path, labels={"a.wav": 0}, alignment={"missing_policy": policy})
    wave(tmp_path / "audio/a.wav")
    wave(tmp_path / "audio/b.wav")
    report, _ = validate_workflow_file(path, check_data=True)
    assert report.valid == valid
    if valid:
        assert report.data_checks["alignments"][0]["missing_count"] == 1
    else:
        assert "data.alignment" in codes(report)


def test_drop_all_samples_fails(tmp_path):
    path = write_workflow(tmp_path, labels={"other.wav": 0}, alignment={"missing_policy": "drop"})
    wave(tmp_path / "audio/a.wav")
    report, _ = validate_workflow_file(path, check_data=True)
    assert "data.empty_alignment" in codes(report)


def test_duplicate_basename_fails(tmp_path):
    path = write_workflow(tmp_path, labels={"a.wav": 0})
    wave(tmp_path / "audio/one/a.wav")
    wave(tmp_path / "audio/two/a.wav")
    report, _ = validate_workflow_file(path, check_data=True)
    assert "data.alignment" in codes(report)


def test_recursive_and_first_n_selection(tmp_path):
    path = write_workflow(tmp_path, parameters={"recursive": False, "max_files": 1})
    wave(tmp_path / "audio/a.wav")
    (tmp_path / "audio/z.wav").write_bytes(b"broken")
    wave(tmp_path / "audio/subfolder/nested.wav")
    report, _ = validate_workflow_file(path, check_data=True)
    assert report.valid
    assert report.data_checks["sources"][0]["candidate_files"] == 2
    assert report.data_checks["sources"][0]["checked_files"] == 1


def test_random_selection_checks_all_candidates(tmp_path):
    path = write_workflow(tmp_path, parameters={"max_files": 1, "selection_mode": "random_n"})
    wave(tmp_path / "audio/a.wav")
    wave(tmp_path / "audio/b.wav")
    report, _ = validate_workflow_file(path, check_data=True)
    assert report.valid
    assert report.data_checks["sources"][0]["checked_files"] == 2
    assert any(w["code"] == "data.random_selection" for w in report.warnings)


def test_invalid_definition_skips_data_check(tmp_path):
    path = tmp_path / "workflow.json"
    path.write_text("{}", encoding="utf-8")
    report, _ = validate_workflow_file(path, check_data=True)
    assert report.data_checks["status"] == "skipped"


def test_audio_file_source_and_json_success(tmp_path, capsys):
    wave(tmp_path / "a.wav")
    path = tmp_path / "workflow.json"
    path.write_text(json.dumps({"nodes": [{"id": "audio", "type": "audio_file", "parameters": {"file_path": "a.wav"}}], "connections": []}), encoding="utf-8")
    assert main(["validate", str(path), "--check-data", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["workflow"]["data_checks"]["model_input_compatibility"] == "not_checked"


def test_invalid_labels_are_validation_errors(tmp_path):
    path = write_workflow(tmp_path, labels={"a.wav": 0})
    wave(tmp_path / "audio/a.wav")
    (tmp_path / "labels.json").write_text("{", encoding="utf-8")
    report, _ = validate_workflow_file(path, check_data=True)
    assert "data.source" in codes(report)


def test_corrupt_audio_cli_stderr_is_still_json(tmp_path, capsys):
    path = write_workflow(tmp_path)
    (tmp_path / "audio/a.wav").write_bytes(b"broken")
    assert main(["validate", str(path), "--check-data", "--json"]) == 3
    assert json.loads(capsys.readouterr().err)["valid"] is False


@pytest.mark.parametrize("mode", ["basename", "stem", "full_path", "relative_path"])
def test_alignment_match_modes_follow_node_contract(tmp_path, mode):
    audio = tmp_path / "audio/a.wav"
    key = {"basename": "a.wav", "stem": "a", "full_path": str(audio), "relative_path": str(audio)}[mode]
    path = write_workflow(tmp_path, labels={key: 1}, alignment={"match_mode": mode})
    wave(audio)
    report, _ = validate_workflow_file(path, check_data=True)
    assert report.valid
    assert report.data_checks["alignments"][0]["matched_count"] == 1


def test_unresolved_alignment_is_explicitly_skipped(tmp_path):
    path = write_workflow(tmp_path, labels={"a.wav": 0})
    wave(tmp_path / "audio/a.wav")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["nodes"].append({"id": "pass", "type": "passthrough", "parameters": {}})
    data["connections"][0]["source"] = {"node": "pass", "port": "out"}
    data["connections"].append({"source": {"node": "source", "port": "file_paths"}, "target": {"node": "pass", "port": "in"}})
    path.write_text(json.dumps(data), encoding="utf-8")
    report, _ = validate_workflow_file(path, check_data=True)
    assert report.valid
    assert report.data_checks["alignments"][0]["status"] == "skipped"
    assert any(w["code"] == "data.alignment_skipped" for w in report.warnings)


def test_empty_target_file_is_rejected(tmp_path):
    path = write_workflow(tmp_path, labels={})
    wave(tmp_path / "audio/a.wav")
    report, _ = validate_workflow_file(path, check_data=True)
    assert "data.source" in codes(report)
