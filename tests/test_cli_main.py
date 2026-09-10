"""Command-level tests for CLI exit codes and artifacts."""

import json

import numpy as np
import soundfile as sf

from src.cli.main import main


def _write_wave(path):
    sf.write(path, np.zeros(800, dtype=np.float32), 8000)


def _audio_file_workflow(path, audio_path):
    path.write_text(
        json.dumps(
            {
                "version": "1.0",
                "metadata": {"name": "CLI smoke test"},
                "nodes": [
                    {
                        "id": "source",
                        "type": "audio_file",
                        "position": [0, 0],
                        "parameters": {"file_path": str(audio_path), "target_sr": 8000},
                    }
                ],
                "connections": [],
            }
        ),
        encoding="utf-8",
    )


def _model_definition(path):
    path.write_text(
        json.dumps(
            {
                "version": "1.0",
                "name": "CLI model smoke test",
                "layers": [
                    {"id": "input", "type": "input", "parameters": {"shape": "(8,)"}},
                    {"id": "dense", "type": "dense", "parameters": {"units": 2}},
                    {
                        "id": "output",
                        "type": "output",
                        "parameters": {
                            "activation": "softmax",
                            "loss": "categorical_crossentropy",
                            "metrics": ["accuracy"],
                        },
                    },
                ],
                "connections": [
                    {"source": "input", "target": "dense"},
                    {"source": "dense", "target": "output"},
                ],
            }
        ),
        encoding="utf-8",
    )


def test_cli_inspect_empty_dataset_has_validation_exit_code(tmp_path, capsys):
    exit_code = main(["inspect", str(tmp_path), "--json"])

    assert exit_code == 3
    assert json.loads(capsys.readouterr().out)["summary"]["valid_files"] == 0


def test_cli_validate_invalid_definition_has_validation_exit_code(tmp_path, capsys):
    workflow = tmp_path / "workflow.json"
    workflow.write_text("{}", encoding="utf-8")

    exit_code = main(["validate", str(workflow), "--json"])

    assert exit_code == 3
    payload = json.loads(capsys.readouterr().err)
    assert payload["valid"] is False


def test_cli_run_writes_manifest_and_jsonl_events(tmp_path, capsys):
    audio = tmp_path / "sample.wav"
    workflow = tmp_path / "workflow.json"
    run_dir = tmp_path / "run"
    _write_wave(audio)
    _audio_file_workflow(workflow, audio)

    exit_code = main(
        ["run", str(workflow), "--run-dir", str(run_dir), "--events", "jsonl"]
    )

    assert exit_code == 0
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    events = [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    stdout_events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]

    assert manifest["status"] == "completed"
    assert manifest["seed"] == 42
    assert (run_dir / "workflow.json").is_file()
    assert events[-1]["event"] == "workflow_execution_finished"
    assert stdout_events[-1]["event"] == "workflow_completed"


def test_cli_run_does_not_overwrite_nonempty_directory(tmp_path, capsys):
    audio = tmp_path / "sample.wav"
    workflow = tmp_path / "workflow.json"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "keep.txt").write_text("keep", encoding="utf-8")
    _write_wave(audio)
    _audio_file_workflow(workflow, audio)

    exit_code = main(["run", str(workflow), "--run-dir", str(run_dir)])

    assert exit_code == 5
    assert (run_dir / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert "not empty" in capsys.readouterr().err


def test_cli_run_refuses_overwrite_for_unmanaged_directory(tmp_path, capsys):
    audio = tmp_path / "sample.wav"
    workflow = tmp_path / "workflow.json"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "keep.txt").write_text("keep", encoding="utf-8")
    _write_wave(audio)
    _audio_file_workflow(workflow, audio)

    exit_code = main(
        ["run", str(workflow), "--run-dir", str(run_dir), "--overwrite"]
    )

    assert exit_code == 5
    assert (run_dir / "keep.txt").is_file()
    assert "not created by this CLI" in capsys.readouterr().err


def test_cli_run_resolves_relative_input_paths(tmp_path):
    audio = tmp_path / "sample.wav"
    workflow = tmp_path / "workflow.json"
    run_dir = tmp_path / "run"
    _write_wave(audio)
    _audio_file_workflow(workflow, "sample.wav")

    assert main(["run", str(workflow), "--run-dir", str(run_dir)]) == 0

    resolved = json.loads((run_dir / "workflow.resolved.json").read_text(encoding="utf-8"))
    assert resolved["nodes"][0]["parameters"]["file_path"] == str(audio.resolve())


def test_cli_run_can_overwrite_its_own_managed_directory(tmp_path):
    audio = tmp_path / "sample.wav"
    workflow = tmp_path / "workflow.json"
    run_dir = tmp_path / "run"
    _write_wave(audio)
    _audio_file_workflow(workflow, audio)

    assert main(["run", str(workflow), "--run-dir", str(run_dir)]) == 0
    assert main(
        ["run", str(workflow), "--run-dir", str(run_dir), "--overwrite"]
    ) == 0


def test_cli_run_rejects_external_output_path(tmp_path):
    audio = tmp_path / "sample.wav"
    workflow = tmp_path / "workflow.json"
    run_dir = tmp_path / "run"
    external = tmp_path / "external"
    _write_wave(audio)
    workflow.write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "id": "source",
                        "type": "audio_file",
                        "parameters": {"file_path": str(audio), "target_sr": 8000},
                    },
                    {
                        "id": "save",
                        "type": "save_audio",
                        "parameters": {"output_folder": str(external)},
                    },
                ],
                "connections": [
                    {
                        "source": {"node": "source", "port": "audio"},
                        "target": {"node": "save", "port": "audio"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert main(["run", str(workflow), "--run-dir", str(run_dir)]) == 3
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["validation"]["errors"][0]["code"] == "output.outside_run_dir"


def test_cli_build_model_creates_keras_artifact(tmp_path):
    definition = tmp_path / "model.model.json"
    output = tmp_path / "model.keras"
    _model_definition(definition)

    exit_code = main(
        ["build-model", str(definition), "--output", str(output), "--no-compile"]
    )

    assert exit_code == 0
    assert output.is_file()
