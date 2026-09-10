"""Tests for side-effect-free CLI validation."""

import json

from src.cli.validation import validate_model_file, validate_workflow_file


def _write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_validate_workflow_accepts_minimal_audio_source(tmp_path):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"placeholder")
    workflow = _write_json(
        tmp_path / "workflow.json",
        {
            "version": "1.0",
            "nodes": [
                {
                    "id": "source",
                    "type": "audio_file",
                    "position": [0, 0],
                    "parameters": {"file_path": "sample.wav", "target_sr": 16000},
                }
            ],
            "connections": [],
        },
    )

    report, loaded = validate_workflow_file(workflow)

    assert report.valid is True
    assert loaded is not None


def test_validate_workflow_rejects_unknown_node_and_parameter(tmp_path):
    workflow = _write_json(
        tmp_path / "workflow.json",
        {
            "nodes": [
                {"id": "unknown", "type": "missing", "parameters": {}},
                {"id": "source", "type": "audio_folder", "parameters": {"surprise": 1}},
            ],
            "connections": [],
        },
    )

    report, loaded = validate_workflow_file(workflow)

    assert loaded is None
    assert {item["code"] for item in report.errors} >= {
        "node.unknown_type",
        "node.unknown_parameter",
    }


def test_validate_workflow_rejects_enabled_breakpoint_in_headless_mode(tmp_path):
    workflow = _write_json(
        tmp_path / "workflow.json",
        {
            "nodes": [{"id": "break", "type": "breakpoint", "parameters": {"enabled": True}}],
            "connections": [],
        },
    )

    report, _ = validate_workflow_file(workflow, headless=True)

    assert any(item["code"] == "headless.breakpoint" for item in report.errors)


def test_validate_workflow_rejects_connection_to_unknown_port(tmp_path):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"placeholder")
    workflow = _write_json(
        tmp_path / "workflow.json",
        {
            "nodes": [
                {"id": "source", "type": "audio_file", "parameters": {"file_path": str(audio)}},
                {"id": "sink", "type": "passthrough", "parameters": {}},
            ],
            "connections": [
                {
                    "source": {"node": "source", "port": "missing"},
                    "target": {"node": "sink", "port": "in"},
                }
            ],
        },
    )

    report, _ = validate_workflow_file(workflow)

    assert any(item["code"] == "connection.unknown_port" for item in report.errors)


def test_validate_model_rejects_unknown_layer(tmp_path):
    model = _write_json(
        tmp_path / "model.model.json",
        {"layers": [{"id": "mystery", "type": "unknown", "parameters": {}}], "connections": []},
    )

    report, graph = validate_model_file(model)

    assert report.valid is False
    assert graph is None
    assert report.errors[0]["code"] == "layer.unknown_type"
