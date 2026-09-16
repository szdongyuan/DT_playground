"""Headless workflow execution and run-manifest handling."""

from __future__ import annotations

import json
import hashlib
import importlib.metadata
import os
import platform
import random
import re
import shutil
import signal
import sys
import tempfile
import time
import uuid
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.cli.contracts import CLI_SCHEMA_VERSION, EventWriter, json_safe, utc_now
from src.cli.output import isolated_event_output
from src.cli.run_lock import RunDirectoryBusyError, run_directory_lock
from src.cli.validation import validate_workflow_file
from src.core.event_bus import get_event_bus
from src.workflow.engine import WorkflowEngine
from src.workflow.path_resolver import resolve_workflow_input_paths


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
    run_id = _prepare_run_directory(target_dir, overwrite)

    started_at = utc_now()
    start_time = time.monotonic()
    report, workflow = validate_workflow_file(source_path, headless=True)
    if report.valid and workflow is not None:
        _resolve_workflow_paths(workflow, source_path.parent, target_dir, report)
    if not report.valid or workflow is None:
        manifest = _manifest_base(source_path, target_dir, seed, started_at, run_id)
        manifest.update(
            {
                "status": "validation_failed",
                "validation": _manifest_safe(
                    report.to_dict(), target_dir, [source_path.parent]
                ),
            }
        )
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
        def emit_node_progress(node_id: str, progress: float, data: str) -> None:
            payload = _parse_progress_data(data)
            emit(
                "node_progress",
                _progress_message(node_id, progress, payload),
                node_id=node_id,
                progress=progress,
                payload=payload,
            )

        engine.node_progress.connect(emit_node_progress)
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

        output_summary = _publish_result_artifacts(
            result.node_results,
            target_dir,
            workflow,
        )
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
        manifest = _manifest_base(source_path, target_dir, seed, started_at, run_id)
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
                "validation": _manifest_safe(
                    report.to_dict(), target_dir, _dataset_roots(workflow)
                ),
                "node_outputs": output_summary,
                "artifacts": {
                    "workflow": "workflow.json",
                    "resolved_workflow": "workflow.resolved.json",
                    "events": "events.jsonl",
                    "manifest": "manifest.json",
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
    manifest_path = target_dir / "manifest.json"
    completion_message = f"{result.message}. Manifest: {manifest_path}"
    writer.emit(
        "workflow_completed" if result.success else "workflow_failed",
        completion_message, success=result.success,
        execution_time_seconds=result.execution_time,
        manifest=str(manifest_path),
    )
    return result.success, manifest


def _prepare_run_directory(path: Path, overwrite: bool) -> str:
    marker_path = path / ".audio-platform-run.json"
    run_id = None
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Run directory is not empty: {path}")
        run_id = _managed_run_id(path, marker_path)
        for child in path.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    path.mkdir(parents=True, exist_ok=True)
    run_id = run_id or str(uuid.uuid4())
    _write_json(
        marker_path,
        {
            "schema_version": "1.0",
            "managed_by": "audio-platform-cli",
            "kind": "run-directory",
            "run_id": run_id,
        },
    )
    return run_id


def _managed_run_id(path: Path, marker_path: Path) -> str:
    """Validate a current marker or a legacy manifest before overwrite."""
    if marker_path.is_file():
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FileExistsError(f"Run directory has an invalid marker: {path}") from exc
        if (
            marker.get("managed_by") != "audio-platform-cli"
            or marker.get("kind") != "run-directory"
            or not marker.get("run_id")
        ):
            raise FileExistsError(f"Run directory marker is incompatible: {path}")
        return str(marker["run_id"])

    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise FileExistsError(
            f"Refusing to overwrite a directory not created by this CLI: {path}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FileExistsError(f"Run directory has an invalid manifest: {path}") from exc
    stored_run_dir = str(manifest.get("run_dir", ""))
    compatible_path = stored_run_dir == "." or (
        stored_run_dir and Path(stored_run_dir).resolve() == path.resolve()
    )
    if (
        manifest.get("schema_version") != CLI_SCHEMA_VERSION
        or manifest.get("managed_by") != "audio-platform-cli"
        or not compatible_path
    ):
        raise FileExistsError(f"Run directory manifest is incompatible: {path}")
    return str(manifest.get("run_id") or uuid.uuid4())


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


def _manifest_base(
    source: Path,
    run_dir: Path,
    seed: int,
    started_at: str,
    run_id: str,
) -> dict[str, Any]:
    return {
        "schema_version": CLI_SCHEMA_VERSION,
        "manifest_version": "1.1",
        "managed_by": "audio-platform-cli",
        "run_id": run_id,
        "started_at": started_at,
        "workflow_source": source.name,
        "workflow_sha256": (
            hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None
        ),
        "run_dir": ".",
        "path_policy": "relative",
        "seed": seed,
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": _package_versions(),
        },
    }


def _publish_result_artifacts(
    node_results: dict[str, dict[str, Any]],
    run_dir: Path,
    workflow,
) -> dict[str, Any]:
    """Publish complete tabular results and return bounded manifest summaries."""
    roots = _dataset_roots(workflow)
    summaries = {}
    for node_id, outputs in node_results.items():
        node_summary = {}
        for name, value in outputs.items():
            if _is_classification_result(value):
                node_summary[name] = _publish_classification_result(
                    node_id,
                    value,
                    run_dir,
                    roots,
                )
            else:
                node_summary[name] = _manifest_safe(value, run_dir, roots)
        summaries[node_id] = node_summary
    return summaries


def _is_classification_result(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("task") == "classification"
        and isinstance(value.get("predictions"), list)
    )


def _publish_classification_result(
    node_id: str,
    result: dict[str, Any],
    run_dir: Path,
    roots: list[Path],
) -> dict[str, Any]:
    results_dir = run_dir / "results"
    artifact_stem = _artifact_stem(node_id)
    predictions_path = results_dir / f"{artifact_stem}.predictions.jsonl"
    errors_path = results_dir / f"{artifact_stem}.misclassified.jsonl"
    rows = []
    errors = []
    for prediction in result.get("predictions", []):
        row = dict(prediction)
        source_ref = _portable_path(row.pop("file_path", None), run_dir, roots)
        row["source_ref"] = source_ref
        row["sample_id"] = _sample_id(source_ref, row.get("index"))
        row["is_error"] = row.get("true_class_id") != row.get("predicted_class_id")
        rows.append(row)
        if row["is_error"]:
            errors.append(row)
    _write_jsonl(predictions_path, rows)
    _write_jsonl(errors_path, errors)
    return {
        "type": "classification_result",
        "schema_version": result.get("schema_version", "1.0"),
        "sample_count": result.get("sample_count", len(rows)),
        "error_count": len(errors),
        "classes": _manifest_safe(result.get("classes", []), run_dir, roots),
        "metrics": _manifest_safe(result.get("metrics", {}), run_dir, roots),
        "confusion_matrix": _manifest_safe(
            result.get("confusion_matrix", []), run_dir, roots
        ),
        "per_class": _manifest_safe(result.get("per_class", []), run_dir, roots),
        "warnings": _manifest_safe(result.get("warnings", []), run_dir, roots),
        "artifacts": {
            "predictions": str(predictions_path.relative_to(run_dir)),
            "misclassified": str(errors_path.relative_to(run_dir)),
        },
    }


def _manifest_safe(
    value: Any,
    run_dir: Path,
    roots: list[Path],
    *,
    depth: int = 0,
) -> Any:
    """Return a bounded, path-redacted summary for manifest publication."""
    if depth > 5:
        return {"type": type(value).__name__}
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, (str, Path)):
        return _portable_path(value, run_dir, roots)
    if isinstance(value, dict):
        if "filename_map" in value and isinstance(value["filename_map"], dict):
            reduced = {key: item for key, item in value.items() if key != "filename_map"}
            reduced["filename_map"] = {
                "type": "mapping",
                "length": len(value["filename_map"]),
            }
            return _manifest_safe(reduced, run_dir, roots, depth=depth + 1)
        items = list(value.items())
        if len(items) > 50:
            return {
                "type": "dict",
                "length": len(items),
                "keys": [
                    str(_portable_path(key, run_dir, roots))
                    for key, _ in items[:10]
                ],
            }
        return {
            str(_portable_path(key, run_dir, roots)): _manifest_safe(
                item, run_dir, roots, depth=depth + 1
            )
            for key, item in items
        }
    if isinstance(value, (list, tuple)):
        if len(value) <= 20:
            return [
                _manifest_safe(item, run_dir, roots, depth=depth + 1)
                for item in value
            ]
        return {
            "type": type(value).__name__,
            "length": len(value),
            "preview": [
                _manifest_safe(item, run_dir, roots, depth=depth + 1)
                for item in value[:3]
            ],
        }
    if is_dataclass(value):
        return {
            item.name: _manifest_safe(
                getattr(value, item.name), run_dir, roots, depth=depth + 1
            )
            for item in fields(value)
        }
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {
            "type": type(value).__name__,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if type(value).__module__ == "numpy" and hasattr(value, "item"):
        return _manifest_safe(value.item(), run_dir, roots, depth=depth + 1)
    return json_safe(value)


def _dataset_roots(workflow) -> list[Path]:
    roots = []
    for node in workflow.nodes.values():
        if node.node_type == "audio_folder":
            value = node.get_parameter("folder_path")
            if value:
                roots.append(Path(value).expanduser().resolve())
        elif node.node_type == "sqlite_audio_database":
            value = node.get_parameter("audio_root")
            if value:
                roots.append(Path(value).expanduser().resolve())
        elif node.node_type == "audio_file":
            value = node.get_parameter("file_path")
            if value:
                roots.append(Path(value).expanduser().resolve().parent)
    return list(dict.fromkeys(roots))


def _portable_path(value: Any, run_dir: Path, roots: list[Path]) -> Any:
    if value is None:
        return None
    text = str(value)
    path = Path(text).expanduser()
    if not path.is_absolute():
        return text.replace("\\", "/")
    resolved = path.resolve()
    try:
        return resolved.relative_to(run_dir.resolve()).as_posix()
    except ValueError:
        pass
    for index, root in enumerate(roots):
        try:
            relative = resolved.relative_to(root)
            prefix = "dataset" if len(roots) == 1 else f"dataset-{index + 1}"
            return f"{prefix}://{relative.as_posix()}"
        except ValueError:
            continue
    return f"external://{resolved.name}"


def _sample_id(source_ref: Any, index: Any) -> str:
    identity = str(source_ref) if source_ref else f"index:{index}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def _artifact_stem(node_id: str) -> str:
    """Return a confined, collision-resistant filename stem for a node ID."""
    text = str(node_id)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._")[:80]
    if safe == text and safe:
        return safe
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]
    return f"{safe or 'node'}-{digest}"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _parse_progress_data(value: str) -> Any:
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value


def _progress_message(node_id: str, progress: float, payload: Any) -> str:
    """Build a compact text message while retaining the full event payload."""
    if isinstance(payload, dict) and payload.get("epoch") is not None:
        epoch = payload["epoch"]
        total = payload.get("total_epochs", "?")
        metrics = []
        for name in ("loss", "accuracy", "val_loss", "val_accuracy"):
            value = payload.get(name)
            if isinstance(value, (int, float)):
                metrics.append(f"{name}={value:.4f}")
        suffix = f": {', '.join(metrics)}" if metrics else ""
        return f"Epoch {epoch}/{total}{suffix}"
    return f"Node progress: {node_id} ({progress * 100:.0f}%)"


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
    resolve_workflow_input_paths(workflow, definition_dir)
    for node in workflow.nodes.values():
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
    except ValueError:
        report.error(
            "output.outside_run_dir",
            "Output path must stay inside the run directory.",
            node_id=node.node_id,
            parameter=parameter,
            reference=str(value),
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
