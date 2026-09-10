"""Headless workflow execution and run-manifest handling."""

from __future__ import annotations

import json
import hashlib
import importlib.metadata
import os
import platform
import random
import shutil
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.cli.contracts import CLI_SCHEMA_VERSION, EventWriter, json_safe, utc_now
from src.cli.output import isolated_event_output
from src.cli.run_lock import RunDirectoryBusyError, run_directory_lock
from src.cli.validation import INPUT_PATH_PARAMETERS, validate_workflow_file
from src.core.event_bus import get_event_bus
from src.workflow.engine import WorkflowEngine


def execute_workflow(
    workflow_path: str | Path,
    run_dir: str | Path,
    *,
    writer: EventWriter,
    seed: int = 42,
    overwrite: bool = False,
    checkpoint: str | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Validate and synchronously execute one workflow in a managed run directory."""
    with isolated_event_output(writer):
        try:
            with run_directory_lock(run_dir) as target_dir:
                return _execute_workflow(
                    workflow_path, target_dir, writer=writer, seed=seed,
                    overwrite=overwrite, checkpoint=checkpoint,
                )
        except RunDirectoryBusyError as exc:
            writer.emit("workflow_failed", str(exc), success=False,
                        phase="directory_lock", code="run_dir.busy")
            raise


def _execute_workflow(
    workflow_path, run_dir, *, writer, seed, overwrite, checkpoint,
) -> tuple[bool, dict[str, Any]]:
    """Execute with output routing already established by the public wrapper."""
    source_path = Path(workflow_path).expanduser().resolve()
    target_dir = Path(run_dir).expanduser().resolve()
    checkpoint_path = _resolve_output_path(checkpoint, target_dir) if checkpoint else None
    if _is_within(source_path, target_dir):
        raise ValueError("Workflow source must be outside the managed run directory.")
    _prepare_run_directory(target_dir, overwrite)

    started_at = utc_now()
    start_time = time.monotonic()
    report, workflow = validate_workflow_file(source_path, headless=True)
    if report.valid and workflow is not None:
        _resolve_workflow_paths(workflow, source_path.parent, target_dir, report)
    if not report.valid or workflow is None:
        manifest = _manifest_base(source_path, target_dir, seed, started_at)
        manifest.update({"status": "validation_failed", "validation": report.to_dict()})
        _write_json(target_dir / "manifest.json", manifest)
        writer.emit("validation_failed", "Workflow validation failed.", errors=report.errors)
        return False, manifest

    shutil.copy2(source_path, target_dir / "workflow.json")
    _write_json(target_dir / "workflow.resolved.json", workflow.to_dict())
    _set_random_seeds(seed, workflow)
    engine = WorkflowEngine()
    engine.set_workflow(workflow)
    events_path = target_dir / "events.jsonl"

    with events_path.open("w", encoding="utf-8") as event_file:
        file_writer = EventWriter("jsonl", event_file)

        def emit(event: str, message: str = "", **data: Any) -> None:
            writer.emit(event, message, **data)
            file_writer.emit(event, message, **data)

        engine.workflow_started.connect(lambda: emit("workflow_started", "Workflow started."))
        engine.progress_updated.connect(
            lambda current, total, message: emit(
                "workflow_progress", message, current=current, total=total
            )
        )
        engine.node_started.connect(
            lambda node_id: emit("node_started", f"Node started: {node_id}", node_id=node_id)
        )
        engine.node_finished.connect(
            lambda node_id, success: emit(
                "node_finished", f"Node finished: {node_id}", node_id=node_id, success=success
            )
        )
        engine.node_progress.connect(
            lambda node_id, progress, data: emit(
                "node_progress", f"Node progress: {node_id}",
                node_id=node_id, progress=progress, payload=_parse_progress_data(data)
            )
        )
        engine.status_message.connect(lambda message: emit("status", message))

        previous_handler = signal.getsignal(signal.SIGINT)

        def handle_interrupt(_signum, _frame) -> None:
            emit("cancellation_requested", "Cancellation requested.")
            if checkpoint_path:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                get_event_bus().training_stop_with_checkpoint.emit(str(checkpoint_path))
            engine.stop()

        signal.signal(signal.SIGINT, handle_interrupt)
        try:
            result = engine.execute_sync()
        finally:
            signal.signal(signal.SIGINT, previous_handler)

        output_summary = {
            node_id: {name: json_safe(value) for name, value in outputs.items()}
            for node_id, outputs in result.node_results.items()
        }
        status = "completed" if result.success else "failed"
        emit(
            "workflow_execution_finished",
            result.message,
            success=result.success,
            execution_time_seconds=result.execution_time,
        )

    # Seal the execution log before hashing it. Finalization notifications are
    # caller-only, so they cannot invalidate the persisted log's full-file hash.
    try:
        manifest = _manifest_base(source_path, target_dir, seed, started_at)
        produced_files = sorted(
            str(path.relative_to(target_dir))
            for path in target_dir.rglob("*")
            if path.is_file() and path.name != "manifest.json"
        )
        manifest.update(
            {
                "finished_at": utc_now(),
                "wall_time_seconds": time.monotonic() - start_time,
                "status": status,
                "success": result.success,
                "message": result.message,
                "execution_time_seconds": result.execution_time,
                "validation": report.to_dict(),
                "node_outputs": output_summary,
                "artifacts": {
                    "workflow": str(target_dir / "workflow.json"),
                    "resolved_workflow": str(target_dir / "workflow.resolved.json"),
                    "events": str(events_path),
                    "manifest": str(target_dir / "manifest.json"),
                    "produced_files": produced_files,
                    "sha256": {
                        name: _file_sha256(target_dir / name) for name in produced_files
                    },
                },
            }
        )
        _write_json(target_dir / "manifest.json", manifest)
    except Exception as exc:
        # Disk or pipe failure must not mask the original finalization error.
        try:
            writer.emit("workflow_failed", str(exc), success=False,
                        phase="finalization", error_type=type(exc).__name__)
        except (OSError, ValueError):
            pass
        raise
    writer.emit(
        "workflow_completed" if result.success else "workflow_failed",
        result.message, success=result.success,
        execution_time_seconds=result.execution_time,
        manifest=str(target_dir / "manifest.json"),
    )
    return result.success, manifest


def _prepare_run_directory(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Run directory is not empty: {path}")
        manifest_path = path / "manifest.json"
        if not manifest_path.is_file():
            raise FileExistsError(
                f"Refusing to overwrite a directory not created by this CLI: {path}"
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FileExistsError(f"Run directory has an invalid manifest: {path}") from exc
        if (
            manifest.get("schema_version") != CLI_SCHEMA_VERSION
            or manifest.get("managed_by") != "audio-platform-cli"
            or Path(manifest.get("run_dir", "")).resolve() != path.resolve()
        ):
            raise FileExistsError(f"Run directory manifest is incompatible: {path}")
        for child in path.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    path.mkdir(parents=True, exist_ok=True)


def _set_random_seeds(seed: int, workflow) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    requires_tensorflow = any(
        node.node_type in {
            "load_model",
            "save_model",
            "classification_trainer",
            "regression_trainer",
            "classification_evaluator",
            "regression_evaluator",
            "classification_predict",
            "regression_predict",
            "ai_feature_extraction",
            "grad_cam",
        }
        for node in workflow.nodes.values()
    )
    if not requires_tensorflow:
        return
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
    except ImportError:
        pass


def _manifest_base(source: Path, run_dir: Path, seed: int, started_at: str) -> dict[str, Any]:
    return {
        "schema_version": CLI_SCHEMA_VERSION,
        "managed_by": "audio-platform-cli",
        "started_at": started_at,
        "workflow_source": str(source),
        "workflow_sha256": (
            hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None
        ),
        "run_dir": str(run_dir),
        "seed": seed,
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": _package_versions(),
        },
    }


def _parse_progress_data(value: str) -> Any:
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Atomically publish JSON without exposing a partially written destination."""
    payload = json.dumps(json_safe(data), ensure_ascii=False, indent=2)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _package_versions() -> dict[str, str]:
    versions = {}
    for distribution in ("tensorflow", "numpy", "scipy", "joblib", "librosa", "scikit-learn", "PySide6"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_workflow_paths(workflow, definition_dir: Path, run_dir: Path, report) -> None:
    for node in workflow.nodes.values():
        for name in INPUT_PATH_PARAMETERS.get(node.node_type, ()):
            value = node.get_parameter(name)
            if not value:
                continue
            path = Path(value).expanduser()
            resolved = path.resolve() if path.is_absolute() else (definition_dir / path).resolve()
            node.parameter_values[name] = str(resolved)

        if node.node_type == "save_audio":
            _set_confined_output(node, "output_folder", run_dir, report)
        elif node.node_type in {"save_anomaly_model", "export_anomaly_results"}:
            _set_confined_output(node, "output_folder", run_dir, report)
            try:
                from src.workflow.anomaly_io import output_target

                name = "file_name" if node.node_type == "save_anomaly_model" else "directory_name"
                target = output_target(node.get_parameter("output_folder"), node.get_parameter(name))
                _resolve_output_path(target, run_dir)
            except ValueError as exc:
                report.error("output.invalid_target", str(exc), node_id=node.node_id)
        elif node.node_type == "save_model":
            parameter = "existing_file" if node.get_parameter("save_mode") == "overwrite" else "save_dir"
            _set_confined_output(node, parameter, run_dir, report)


def _set_confined_output(node, parameter: str, run_dir: Path, report) -> None:
    value = node.get_parameter(parameter)
    if not value:
        return
    try:
        resolved = _resolve_output_path(value, run_dir)
    except ValueError as exc:
        report.error(
            "output.outside_run_dir",
            str(exc),
            node_id=node.node_id,
            parameter=parameter,
        )
        return
    node.parameter_values[parameter] = str(resolved)


def _resolve_output_path(value: str | Path, run_dir: Path) -> Path:
    path = Path(value).expanduser()
    resolved = path.resolve() if path.is_absolute() else (run_dir / path).resolve()
    try:
        resolved.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"Output path must stay inside the run directory: {resolved}") from exc
    return resolved


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False
