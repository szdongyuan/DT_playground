"""Real process ownership tests for BUG2026091002."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest
import soundfile as sf

from src.cli.run_lock import run_directory_lock


ROOT = Path(__file__).resolve().parents[1]
HOLDER = r'''
import sys, time
from pathlib import Path
from src.cli import runner
from src.cli.main import main
stage, folder, workflow, run = sys.argv[1:]
folder = Path(folder)
def gate():
    (folder / "ready").touch()
    end = time.monotonic() + 60
    while not (folder / "release").exists():
        if time.monotonic() > end:
            raise TimeoutError("test owner gate timed out")
        time.sleep(.01)
prepare = runner._prepare_run_directory
write = runner._write_json
emit = runner.EventWriter.emit
def prepare_probe(*args, **kwargs):
    if stage == "prepare": gate()
    return prepare(*args, **kwargs)
def write_probe(path, data):
    if stage == "manifest" and path.name == "manifest.json": gate()
    return write(path, data)
def emit_probe(self, event, *args, **kwargs):
    if stage == "notification" and event == "workflow_completed": gate()
    return emit(self, event, *args, **kwargs)
runner._prepare_run_directory = prepare_probe
runner._write_json = write_probe
runner.EventWriter.emit = emit_probe
raise SystemExit(main(["run", workflow, "--run-dir", run, "--events", "jsonl"]))
'''


@pytest.fixture
def workflow(tmp_path):
    audio = tmp_path / "sample.wav"
    sf.write(audio, np.zeros(800, dtype=np.float32), 8000)
    path = tmp_path / "workflow.json"
    path.write_text(json.dumps({"nodes": [{"id": "source", "type": "audio_file",
        "parameters": {"file_path": str(audio), "target_sr": 8000}}],
        "connections": []}), encoding="utf-8")
    return path


def invoke(workflow, run, *extra):
    return subprocess.run([sys.executable, str(ROOT / "cli_main.py"), "run", str(workflow),
        "--run-dir", str(run), "--events", "jsonl", *extra], cwd=ROOT,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        capture_output=True, text=True, encoding="utf-8", timeout=60)


@contextmanager
def owner(tmp_path, workflow, stage):
    with (tmp_path / "owner.out").open("w", encoding="utf-8") as out, (tmp_path / "owner.err").open("w", encoding="utf-8") as err:
        process = subprocess.Popen([sys.executable, "-c", HOLDER, stage, str(tmp_path),
            str(workflow), str(tmp_path / "run")], cwd=ROOT, stdout=out, stderr=err,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        try:
            end = time.monotonic() + 60
            while not (tmp_path / "ready").exists():
                assert process.poll() is None, (tmp_path / "owner.err").read_text(encoding="utf-8")
                assert time.monotonic() < end, "Owner failed to reach gate"
                time.sleep(.02)
            yield process
        finally:
            (tmp_path / "release").touch()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


@pytest.mark.parametrize("stage", ["prepare", "manifest", "notification"])
@pytest.mark.parametrize("overwrite", [False, True])
def test_owner_blocks_competitor_through_final_notification(tmp_path, workflow, stage, overwrite):
    run = tmp_path / "run"
    with owner(tmp_path, workflow, stage) as process:
        before = {str(p.relative_to(run)): p.read_bytes() for p in run.rglob("*") if p.is_file()}
        result = invoke(workflow, run, *(["--overwrite"] if overwrite else []))
        assert result.returncode == 5, result.stderr
        events = [json.loads(line) for line in result.stdout.splitlines()]
        assert len(events) == 1
        assert events[0]["data"]["code"] == "run_dir.busy"
        after = {str(p.relative_to(run)): p.read_bytes() for p in run.rglob("*") if p.is_file()}
        assert before == after
    assert process.returncode == 0
    assert json.loads((run / "manifest.json").read_text(encoding="utf-8"))["success"]


def test_independent_directory_and_canonical_alias(tmp_path, workflow):
    with owner(tmp_path, workflow, "prepare") as process:
        alias = tmp_path / "unused" / ".." / "run"
        assert invoke(workflow, alias).returncode == 5
        assert invoke(workflow, tmp_path / "different").returncode == 0
    assert process.returncode == 0


def test_killed_owner_releases_lock_without_deleting_sidecar(tmp_path, workflow):
    with owner(tmp_path, workflow, "prepare") as process:
        sidecars = list(tmp_path.glob(".audio-platform-cli-*.lock"))
        assert len(sidecars) == 1
        process.kill()
        process.wait(timeout=10)
        assert invoke(workflow, tmp_path / "run").returncode == 0
        assert sidecars[0].exists()


def test_exception_releases_lock(tmp_path):
    run = tmp_path / "run"
    with pytest.raises(RuntimeError):
        with run_directory_lock(run):
            raise RuntimeError("test exception")
    with run_directory_lock(run):
        pass


def test_successful_run_can_be_overwritten_after_lock_release(tmp_path, workflow):
    run = tmp_path / "run"
    assert invoke(workflow, run).returncode == 0
    assert invoke(workflow, run, "--overwrite").returncode == 0
    assert len(list(tmp_path.glob(".audio-platform-cli-*.lock"))) == 1


def test_killed_owner_keeps_partial_artifact_protection(tmp_path, workflow):
    with owner(tmp_path, workflow, "manifest") as process:
        process.kill()
        process.wait(timeout=10)
        run = tmp_path / "run"
        before = {p.name: p.read_bytes() for p in run.iterdir() if p.is_file()}
        result = invoke(workflow, run, "--overwrite")
        assert result.returncode == 5
        assert "already in use" not in result.stderr
        assert "not created by this CLI" in result.stderr
        assert before == {p.name: p.read_bytes() for p in run.iterdir() if p.is_file()}


def test_validation_failure_releases_lock(tmp_path, workflow):
    bad = tmp_path / "invalid.json"
    bad.write_text("{}", encoding="utf-8")
    run = tmp_path / "run"
    assert invoke(bad, run).returncode == 3
    assert invoke(workflow, run, "--overwrite").returncode == 0
