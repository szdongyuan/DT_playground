"""Regression coverage for training progress contaminating JSONL stdout."""

import io
import json
import logging
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

from src.cli.contracts import EventWriter
from src.cli.output import isolated_event_output
from src.utils.runtime import get_training_verbose


@pytest.mark.parametrize("failure", [None, RuntimeError, KeyboardInterrupt])
def test_output_isolation_and_restoration(monkeypatch, failure):
    stdout, stderr = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    writer = EventWriter("jsonl")
    logger = logging.getLogger("jsonl-routing-test")
    handler = logging.StreamHandler(stdout)
    logger.addHandler(handler)
    try:
        try:
            with isolated_event_output(writer):
                assert get_training_verbose() == 0
                print("plain output")
                logger.warning("ordinary log")
                writer.emit("test")
                if failure:
                    raise failure("expected")
        except (RuntimeError, KeyboardInterrupt):
            assert failure is not None
        assert sys.stdout is stdout
        assert writer.stream is stdout
        assert handler.stream is stdout
        assert get_training_verbose() == 1
        assert json.loads(stdout.getvalue())["event"] == "test"
        assert "plain output" in stderr.getvalue()
        assert "ordinary log" in stderr.getvalue()
    finally:
        logger.removeHandler(handler)


def test_text_mode_does_not_change_defaults(monkeypatch):
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    original = sys.stdout
    with isolated_event_output(EventWriter("text")):
        assert sys.stdout is original
        assert get_training_verbose() == 1
    monkeypatch.setattr(sys, "frozen", True)
    assert get_training_verbose() == 0


def test_low_level_output_isolated_in_subprocess():
    code = '''
import os
from src.cli.contracts import EventWriter
from src.cli.output import isolated_event_output
writer = EventWriter("jsonl")
with isolated_event_output(writer):
    os.write(1, b"native diagnostic\\n")
    print("python diagnostic")
    writer.emit("inside")
writer.emit("restored")
'''
    result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert [json.loads(line)["event"] for line in result.stdout.splitlines()] == ["inside", "restored"]
    assert "native diagnostic" in result.stderr
    assert "python diagnostic" in result.stderr


def test_real_training_stdout_is_jsonl(tmp_path):
    from tensorflow import keras

    model = keras.Sequential([keras.Input(shape=(1, 80)), keras.layers.Flatten(), keras.layers.Dense(2, activation="softmax")])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.save(tmp_path / "initial.keras")
    for index in range(4):
        folder = tmp_path / "audio" / str(index % 2)
        folder.mkdir(parents=True, exist_ok=True)
        sf.write(folder / f"{index}.wav", np.zeros(80, dtype=np.float32), 8000)
    nodes = [
        {"id": "audio", "type": "audio_folder", "parameters": {"folder_path": "audio", "target_sr": 8000}},
        {"id": "model", "type": "load_model", "parameters": {"model_path": "initial.keras"}},
        {"id": "train", "type": "classification_trainer", "parameters": {"epochs": 1, "batch_size": 2, "early_stopping": False}},
        {"id": "save", "type": "save_model", "parameters": {"save_dir": "models", "file_name": "trained.keras"}},
    ]
    edges = [("audio", "audio", "train", "x_train"), ("audio", "labels", "train", "y_train"), ("model", "model", "train", "model"), ("train", "trained_model", "save", "model")]
    workflow = tmp_path / "workflow.json"
    workflow.write_text(json.dumps({"nodes": nodes, "connections": [
        {"source": {"node": a, "port": b}, "target": {"node": c, "port": d}}
        for a, b, c, d in edges
    ]}), encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "TF_NUM_INTRAOP_THREADS": "1", "TF_NUM_INTEROP_THREADS": "1", "OMP_NUM_THREADS": "1", "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run([sys.executable, str(root / "cli_main.py"), "run", str(workflow), "--run-dir", str(tmp_path / "run"), "--events", "jsonl"], cwd=root, env=env, capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert result.returncode == 0, result.stderr + result.stdout
    assert "\x1b" not in result.stdout
    events = [json.loads(line) for line in result.stdout.splitlines()]
    assert events[-1]["event"] == "workflow_completed"
    progress = [e for e in events if e["event"] == "node_progress" and e["data"]["node_id"] == "train"]
    assert any("loss" in e["data"]["payload"] and "accuracy" in e["data"]["payload"] for e in progress)
    assert (tmp_path / "run/models/trained.keras").is_file()
    disk = [json.loads(line) for line in (tmp_path / "run/events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert disk[-1]["event"] == "workflow_execution_finished"
    assert [(e["event"], e["data"]) for e in events[:-1]] == [(e["event"], e["data"]) for e in disk]
