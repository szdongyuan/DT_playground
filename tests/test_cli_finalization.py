"""Regression tests for BUG2026091001: publish success after finalization."""

import errno
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from src.cli import runner
from src.cli.contracts import EventWriter


def workflow_fixture(folder):
    audio = folder / "sample.wav"
    sf.write(audio, np.zeros(800, dtype=np.float32), 8000)
    path = folder / "workflow.json"
    path.write_text(json.dumps({"nodes": [{"id": "audio", "type": "audio_file",
        "parameters": {"file_path": str(audio), "target_sr": 8000}}],
        "connections": []}), encoding="utf-8")
    return path


@pytest.mark.parametrize("failure", ["none", "permission", "space", "hash"])
def test_subprocess_finalization_contract(tmp_path, failure):
    workflow = workflow_fixture(tmp_path)
    run = tmp_path / "run"
    code = '''
import errno, os, sys
from pathlib import Path
from unittest.mock import patch
from src.cli import runner
from src.cli.main import main
failure = sys.argv[1]
replace = os.replace
hash_file = runner._file_sha256
def replace_probe(source, destination):
    if Path(destination).name == "manifest.json" and failure in {"permission", "space"}:
        code = errno.EACCES if failure == "permission" else errno.ENOSPC
        raise OSError(code, "injected manifest failure")
    return replace(source, destination)
def hash_probe(path):
    if failure == "hash":
        raise OSError(errno.EIO, "injected hash failure")
    return hash_file(path)
with patch.object(os, "replace", replace_probe), patch.object(runner, "_file_sha256", hash_probe):
    raise SystemExit(main(sys.argv[2:]))
'''
    result = subprocess.run(
        [sys.executable, "-c", code, failure, "run", str(workflow),
         "--run-dir", str(run), "--events", "jsonl"],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONIOENCODING": "utf-8", "CUDA_VISIBLE_DEVICES": "-1"},
        capture_output=True, text=True, encoding="utf-8", timeout=120,
    )
    events = [json.loads(line) for line in result.stdout.splitlines()]
    disk_events = [json.loads(line) for line in (run / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert disk_events[-1]["event"] == "workflow_execution_finished"
    assert not any(e["event"] == "workflow_completed" for e in disk_events)
    assert not list(run.glob(".*.tmp"))
    if failure == "none":
        assert result.returncode == 0, result.stderr
        assert events[-1]["event"] == "workflow_completed"
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["status"] == "completed"
        for name, digest in manifest["artifacts"]["sha256"].items():
            assert hashlib.sha256((run / name).read_bytes()).hexdigest() == digest
    else:
        assert result.returncode == 5, result.stderr
        assert not (run / "manifest.json").exists()
        assert not any(e["event"] == "workflow_completed" for e in events)
        assert events[-1]["event"] == "workflow_failed"
        assert events[-1]["data"]["phase"] == "finalization"
        assert events[-1]["data"]["success"] is False
        assert "injected" in result.stderr


def test_success_notification_observes_committed_manifest(tmp_path):
    path = workflow_fixture(tmp_path)
    run = tmp_path / "run"

    class ObservingWriter(EventWriter):
        def emit(self, event, message="", **data):
            if event == "workflow_completed":
                manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
                assert manifest["status"] == "completed"
                digest = manifest["artifacts"]["sha256"]["events.jsonl"]
                assert hashlib.sha256((run / "events.jsonl").read_bytes()).hexdigest() == digest
            return super().emit(event, message, **data)

    success, _ = runner.execute_workflow(path, run, writer=ObservingWriter("jsonl", io.StringIO()))
    assert success


@pytest.mark.parametrize("failure", ["replace", "fsync"])
def test_atomic_json_failure_preserves_previous_destination(tmp_path, failure):
    destination = tmp_path / "manifest.json"
    destination.write_text('{"previous": true}', encoding="utf-8")
    with patch.object(runner.os, failure, side_effect=OSError(errno.ENOSPC, "injected")):
        with pytest.raises(OSError):
            runner._write_json(destination, {"new": True})
    assert json.loads(destination.read_text(encoding="utf-8")) == {"previous": True}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["manifest.json"]


def test_atomic_json_success(tmp_path):
    destination = tmp_path / "manifest.json"
    runner._write_json(destination, {"first": True})
    runner._write_json(destination, {"second": True})
    assert json.loads(destination.read_text(encoding="utf-8")) == {"second": True}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["manifest.json"]


def test_partial_temporary_write_preserves_destination(tmp_path):
    destination = tmp_path / "manifest.json"
    destination.write_text('{"previous": true}', encoding="utf-8")
    original = runner.tempfile.NamedTemporaryFile

    def partial_writer(**kwargs):
        stream = original(**kwargs)

        def fail(payload):
            stream.file.write(payload[:3])
            raise OSError(errno.ENOSPC, "partial temporary write")

        stream.write = fail
        return stream

    with patch.object(runner.tempfile, "NamedTemporaryFile", partial_writer):
        with pytest.raises(OSError, match="partial temporary write"):
            runner._write_json(destination, {"new": True})
    assert json.loads(destination.read_text(encoding="utf-8")) == {"previous": True}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["manifest.json"]


def test_failed_execution_is_not_published_as_success(tmp_path):
    workflow = workflow_fixture(tmp_path)
    run = tmp_path / "run"
    from types import SimpleNamespace

    failed_result = SimpleNamespace(success=False, message="injected node failure",
                                    execution_time=0.1, node_results={})
    stream = io.StringIO()
    with patch.object(runner.WorkflowEngine, "execute_sync", return_value=failed_result):
        success, manifest = runner.execute_workflow(workflow, run, writer=EventWriter("jsonl", stream))
    assert not success and manifest["status"] == "failed"
    events = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert [e["event"] for e in events] == ["workflow_execution_finished", "workflow_failed"]
    assert all(e["data"]["success"] is False for e in events)
