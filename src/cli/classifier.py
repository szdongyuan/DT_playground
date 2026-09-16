"""End-to-end orchestration for CLI audio classification experiments."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.cli.contracts import CLI_SCHEMA_VERSION, EventWriter, utc_now
from src.cli.dataset import inspect_dataset, load_label_mapping
from src.cli.generation import (
    ModelGenerationRequest,
    WorkflowGenerationRequest,
    create_model_definition,
    create_workflow_definition,
    mel_input_shape,
)
from src.cli.runner import execute_workflow
from src.cli.validation import validate_model_file, validate_workflow_file


@dataclass(frozen=True)
class ClassifierTrainingRequest:
    """Inputs for one generated classification experiment."""

    dataset: str | Path
    labels: str | Path
    experiment_dir: str | Path
    template: str = "mel-cnn"
    sample_rate: int = 16000
    duration: float = 3.0
    n_mels: int = 64
    n_fft: int = 1024
    hop_length: int = 512
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    epochs: int = 40
    batch_size: int = 32
    patience: int = 7
    learning_rate: float = 0.001
    dropout: float = 0.4
    model_name: str = "classifier"
    seed: int = 42
    overwrite: bool = False
    checkpoint: str | None = None


def train_classifier(
    request: ClassifierTrainingRequest,
    *,
    writer: EventWriter,
) -> tuple[bool, dict[str, Any]]:
    """Run an experiment and publish a failure event for orchestration errors."""
    try:
        return _train_classifier(request, writer=writer)
    except Exception as exc:
        _record_experiment_failure(request.experiment_dir, exc)
        writer.emit(
            "classification_failed",
            str(exc),
            success=False,
            phase="orchestration",
            error_type=type(exc).__name__,
        )
        raise


def _train_classifier(
    request: ClassifierTrainingRequest,
    *,
    writer: EventWriter,
) -> tuple[bool, dict[str, Any]]:
    """Generate, validate, run, and verify a classification experiment."""
    experiment_dir = Path(request.experiment_dir).expanduser().resolve()
    _prepare_experiment_directory(experiment_dir, request.overwrite)
    experiment_state = {
        "schema_version": CLI_SCHEMA_VERSION,
        "managed_by": "audio-platform-cli",
        "kind": "classification_experiment",
        "status": "preparing",
        "experiment_dir": str(experiment_dir),
        "started_at": utc_now(),
        "seed": request.seed,
    }
    _write_json(experiment_dir / "experiment.json", experiment_state)

    writer.emit("classification_inspection_started", "Inspecting the dataset and labels.")
    profile = inspect_dataset(request.dataset, labels_path=request.labels)
    _validate_profile(profile)
    labels = load_label_mapping(request.labels)
    class_mapping, encoded_labels = _encode_labels(labels)
    profile["external_class_mapping"] = class_mapping
    _write_json(experiment_dir / "dataset-profile.json", profile)
    canonical_labels = experiment_dir / "labels.json"
    _write_json(
        canonical_labels,
        {
            "version": "1.0",
            "labels": encoded_labels,
            "label_names": class_mapping,
        },
    )

    model_definition = experiment_dir / "model.json"
    initial_model = experiment_dir / "initial.keras"
    workflow_definition = experiment_dir / "workflow.json"
    run_dir = experiment_dir / "run-001"
    saved_name = _saved_model_name(request.model_name)
    input_shape = mel_input_shape(
        sample_rate=request.sample_rate,
        duration=request.duration,
        n_mels=request.n_mels,
        hop_length=request.hop_length,
    )

    writer.emit("classification_model_generation_started", "Generating the model definition.")
    model_result = create_model_definition(
        ModelGenerationRequest(
            output=model_definition,
            classes=len(class_mapping),
            template=request.template,
            input_shape=input_shape,
            learning_rate=request.learning_rate,
            dropout=request.dropout,
            name=Path(saved_name).stem,
            overwrite=request.overwrite,
        )
    )
    _build_initial_model(
        model_definition,
        initial_model,
        seed=request.seed,
        overwrite=request.overwrite,
    )

    writer.emit("classification_workflow_generation_started", "Generating the training workflow.")
    workflow_result = create_workflow_definition(
        WorkflowGenerationRequest(
            output=workflow_definition,
            dataset=request.dataset,
            labels=canonical_labels,
            model=initial_model,
            output_model=saved_name,
            sample_rate=request.sample_rate,
            duration=request.duration,
            n_mels=request.n_mels,
            n_fft=request.n_fft,
            hop_length=request.hop_length,
            train_ratio=request.train_ratio,
            val_ratio=request.val_ratio,
            test_ratio=request.test_ratio,
            epochs=request.epochs,
            batch_size=request.batch_size,
            patience=request.patience,
            seed=request.seed,
            name=f"{Path(saved_name).stem}_training",
            overwrite=request.overwrite,
        )
    )

    writer.emit("classification_preflight_started", "Running full dataset preflight checks.")
    preflight, _ = validate_workflow_file(
        workflow_definition,
        headless=True,
        check_data=True,
    )
    if not preflight.valid:
        experiment_state.update(
            {
                "status": "validation_failed",
                "finished_at": utc_now(),
                "validation": preflight.to_dict(),
            }
        )
        _write_json(experiment_dir / "experiment.json", experiment_state)
        payload = _result_payload(
            request,
            experiment_dir,
            class_mapping,
            model_result,
            workflow_result,
            preflight=preflight.to_dict(),
            status="validation_failed",
        )
        writer.emit("classification_failed", payload["summary"], **payload)
        return False, payload

    writer.emit("classification_training_started", "Starting the generated training workflow.")
    success, manifest = execute_workflow(
        workflow_definition,
        run_dir,
        writer=writer,
        seed=request.seed,
        overwrite=request.overwrite,
        checkpoint=request.checkpoint,
    )
    saved_model = run_dir / "models" / saved_name
    verified = success and manifest.get("status") == "completed" and saved_model.is_file()
    status = "completed" if verified else "failed"
    payload = _result_payload(
        request,
        experiment_dir,
        class_mapping,
        model_result,
        workflow_result,
        preflight=preflight.to_dict(),
        status=status,
        manifest=manifest,
        saved_model=saved_model,
    )
    experiment_state.update(
        {
            "status": status,
            "finished_at": utc_now(),
            "result": payload,
        }
    )
    _write_json(experiment_dir / "experiment.json", experiment_state)
    writer.emit(
        "classification_completed" if verified else "classification_failed",
        payload["summary"],
        **payload,
    )
    return verified, payload


def _record_experiment_failure(path_value: str | Path, exc: Exception) -> None:
    """Update an existing trusted experiment marker after a failed step."""
    path = Path(path_value).expanduser().resolve()
    marker = path / "experiment.json"
    if not marker.is_file():
        return
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
        if (
            data.get("managed_by") != "audio-platform-cli"
            or data.get("kind") != "classification_experiment"
            or Path(data.get("experiment_dir", "")).resolve() != path
        ):
            return
        data.update(
            {
                "status": "failed",
                "finished_at": utc_now(),
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }
        )
        _write_json(marker, data)
    except (OSError, ValueError, json.JSONDecodeError):
        return


def _prepare_experiment_directory(path: Path, overwrite: bool) -> None:
    if path.exists() and not path.is_dir():
        raise FileExistsError(f"Experiment path is not a directory: {path}")
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Experiment directory is not empty: {path}")
        marker = path / "experiment.json"
        if not marker.is_file():
            raise FileExistsError(
                f"Refusing to overwrite a directory not created by this CLI: {path}"
            )
        try:
            data = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FileExistsError(f"Experiment directory has an invalid marker: {path}") from exc
        if (
            data.get("schema_version") != CLI_SCHEMA_VERSION
            or data.get("managed_by") != "audio-platform-cli"
            or data.get("kind") != "classification_experiment"
            or Path(data.get("experiment_dir", "")).resolve() != path
        ):
            raise FileExistsError(f"Experiment directory marker is incompatible: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _validate_profile(profile: dict[str, Any]) -> None:
    summary = profile["summary"]
    alignment = profile.get("label_alignment") or {}
    if not summary["valid_files"]:
        raise ValueError("The dataset contains no readable supported audio files.")
    if summary["invalid_files"]:
        raise ValueError("The dataset contains unreadable audio files.")
    for key, message in (
        ("missing_labels", "Some audio files do not have labels."),
        ("orphan_labels", "Some label entries do not match audio files."),
        ("ambiguous_basenames", "Some label basenames match multiple audio files."),
    ):
        if alignment.get(key):
            raise ValueError(message)


def _encode_labels(labels: dict[str, Any]) -> tuple[dict[str, int], dict[str, int]]:
    names_by_text: dict[str, Any] = {}
    for value in labels.values():
        if isinstance(value, (dict, list)) or value is None:
            raise ValueError("Classification labels must be non-null scalar values.")
        text = str(value)
        if not text.strip():
            raise ValueError("Classification labels must not be empty.")
        if text in names_by_text and names_by_text[text] != value:
            raise ValueError(f"Distinct labels have the same text representation: {text}")
        names_by_text[text] = value
    if len(names_by_text) < 2:
        raise ValueError("Classification requires at least two label classes.")
    class_mapping = {name: index for index, name in enumerate(sorted(names_by_text))}
    encoded = {
        name: class_mapping[str(value)]
        for name, value in sorted(labels.items())
    }
    return class_mapping, encoded


def _build_initial_model(
    definition: Path,
    output: Path,
    *,
    seed: int,
    overwrite: bool,
) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(f"Model output already exists: {output}")
    report, graph = validate_model_file(definition)
    if not report.valid or graph is None:
        raise ValueError("Generated model definition failed validation.")
    from tensorflow import keras

    keras.utils.set_random_seed(seed)
    model = graph.build_keras_model(compile_model=True)
    temporary = output.with_name(f".{output.stem}.tmp.keras")
    try:
        if temporary.exists():
            temporary.unlink()
        model.save(str(temporary))
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def _saved_model_name(value: str) -> str:
    name = value.strip()
    if not name or Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError("Model name must be a file name, not a path.")
    if Path(name).suffix and Path(name).suffix.lower() not in {".keras", ".h5"}:
        raise ValueError("Model name must use the .keras or .h5 format.")
    return name if Path(name).suffix else f"{name}.keras"


def _result_payload(
    request: ClassifierTrainingRequest,
    experiment_dir: Path,
    class_mapping: dict[str, int],
    model_result: dict[str, Any],
    workflow_result: dict[str, Any],
    *,
    preflight: dict[str, Any],
    status: str,
    manifest: dict[str, Any] | None = None,
    saved_model: Path | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": CLI_SCHEMA_VERSION,
        "summary": (
            f"Classification experiment completed: {experiment_dir}"
            if status == "completed"
            else f"Classification experiment {status}: {experiment_dir}"
        ),
        "status": status,
        "experiment_dir": str(experiment_dir),
        "dataset_profile": str(experiment_dir / "dataset-profile.json"),
        "labels": str(experiment_dir / "labels.json"),
        "class_count": len(class_mapping),
        "class_mapping": class_mapping,
        "seed": request.seed,
        "model_definition": model_result["output"],
        "initial_model": str(experiment_dir / "initial.keras"),
        "workflow_definition": workflow_result["output"],
        "preflight": preflight,
        "manifest": str(experiment_dir / "run-001" / "manifest.json") if manifest else None,
        "saved_model": str(saved_model) if saved_model else None,
        "run_status": manifest.get("status") if manifest else None,
    }


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
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
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
