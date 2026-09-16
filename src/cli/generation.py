"""Deterministic model and workflow generation for CLI automation."""

from __future__ import annotations

import ast
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.cli.contracts import CLI_SCHEMA_VERSION
from src.cli.validation import validate_model_file, validate_workflow_file


@dataclass(frozen=True)
class ModelGenerationRequest:
    """Inputs for a supported model template."""

    output: str | Path
    classes: int
    task: str = "classification"
    template: str = "mel-cnn"
    input_shape: str = "(64, 94, 1)"
    learning_rate: float = 0.001
    dropout: float = 0.4
    name: str = "mel_cnn_classifier"
    overwrite: bool = False


@dataclass(frozen=True)
class WorkflowGenerationRequest:
    """Inputs for a supported classification workflow."""

    output: str | Path
    dataset: str | Path
    labels: str | Path
    model: str | Path
    output_model: str
    task: str = "classification"
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
    seed: int = 42
    name: str = "classification_training"
    overwrite: bool = False


def create_model_definition(request: ModelGenerationRequest) -> dict[str, Any]:
    """Validate and atomically publish a model definition."""
    _validate_model_request(request)
    output = _output_path(request.output, request.overwrite)
    definition = _mel_cnn_definition(request)
    report = _validate_then_publish(
        output,
        definition,
        lambda path: validate_model_file(path),
    )
    return {
        "schema_version": CLI_SCHEMA_VERSION,
        "summary": f"Model definition created: {output}",
        "task": request.task,
        "template": request.template,
        "output": str(output),
        "classes": request.classes,
        "input_shape": list(_parse_input_shape(request.input_shape)),
        "validation": report.to_dict(),
    }


def create_workflow_definition(request: WorkflowGenerationRequest) -> dict[str, Any]:
    """Validate and atomically publish a classification workflow."""
    normalized = _validate_workflow_request(request)
    output = _output_path(request.output, request.overwrite)
    definition = _classification_workflow(request, **normalized)
    report = _validate_then_publish(
        output,
        definition,
        lambda path: validate_workflow_file(path, headless=True),
    )
    return {
        "schema_version": CLI_SCHEMA_VERSION,
        "summary": f"Workflow definition created: {output}",
        "task": request.task,
        "output": str(output),
        "dataset": str(normalized["dataset"]),
        "labels": str(normalized["labels"]),
        "model": str(normalized["model"]),
        "saved_model": f"models/{normalized['output_model']}",
        "seed": request.seed,
        "validation": report.to_dict(),
    }


def mel_input_shape(
    *, sample_rate: int, duration: float, n_mels: int, hop_length: int
) -> str:
    """Return the channels-last Mel shape produced by the current extractor."""
    frames = 1 + int(sample_rate * duration) // hop_length
    return f"({n_mels}, {frames}, 1)"


def _validate_model_request(request: ModelGenerationRequest) -> None:
    if request.task != "classification":
        raise ValueError("Phase 1 supports only the classification task.")
    if request.template != "mel-cnn":
        raise ValueError("Phase 1 supports only the mel-cnn template.")
    if request.classes < 2:
        raise ValueError("Classification requires at least two classes.")
    _parse_input_shape(request.input_shape)
    if not 0.0 <= request.dropout < 1.0:
        raise ValueError("Dropout must be at least 0 and less than 1.")
    if not 0.000001 <= request.learning_rate <= 1.0:
        raise ValueError("Learning rate must be between 0.000001 and 1.0.")
    if not request.name.strip():
        raise ValueError("Model name must not be empty.")


def _validate_workflow_request(request: WorkflowGenerationRequest) -> dict[str, Any]:
    if request.task != "classification":
        raise ValueError("Phase 1 supports only the classification task.")
    dataset = Path(request.dataset).expanduser().resolve()
    labels = Path(request.labels).expanduser().resolve()
    model = Path(request.model).expanduser().resolve()
    if not dataset.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset}")
    if not labels.is_file():
        raise FileNotFoundError(f"Label file does not exist: {labels}")
    if labels.suffix.lower() not in {".csv", ".json"}:
        raise ValueError("Classification labels must be a CSV or JSON file.")
    if not model.is_file():
        raise FileNotFoundError(f"Model artifact does not exist: {model}")
    if model.suffix.lower() not in {".keras", ".h5"}:
        raise ValueError("Model artifact must use the .keras or .h5 format.")
    if not 8000 <= request.sample_rate <= 48000:
        raise ValueError("Sample rate must be between 8000 and 48000.")
    if request.duration <= 0:
        raise ValueError("Duration must be greater than 0.")
    if not 20 <= request.n_mels <= 256:
        raise ValueError("Mel filters must be between 20 and 256.")
    if not 256 <= request.n_fft <= 8192:
        raise ValueError("FFT window size must be between 256 and 8192.")
    if not 64 <= request.hop_length <= 2048:
        raise ValueError("Hop length must be between 64 and 2048.")
    ratios = (request.train_ratio, request.val_ratio, request.test_ratio)
    if any(value <= 0 or value >= 1 for value in ratios):
        raise ValueError("Train, validation, and test ratios must be between 0 and 1.")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("Train, validation, and test ratios must sum to 1.")
    if request.epochs < 1 or request.batch_size < 1 or request.patience < 1:
        raise ValueError("Epochs, batch size, and patience must be positive integers.")
    output_model = _normalize_model_name(request.output_model)
    return {
        "dataset": dataset,
        "labels": labels,
        "model": model,
        "output_model": output_model,
    }


def _parse_input_shape(value: str) -> tuple[int, int, int]:
    try:
        shape = ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise ValueError("Input shape must be a tuple such as (64, 94, 1).") from exc
    if (
        not isinstance(shape, tuple)
        or len(shape) != 3
        or any(not isinstance(item, int) or item <= 0 for item in shape)
    ):
        raise ValueError("Mel-CNN input shape must contain three positive integers.")
    return shape


def _normalize_model_name(value: str) -> str:
    name = value.strip()
    if not name or Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError("Output model must be a file name, not a path.")
    if Path(name).suffix:
        if Path(name).suffix.lower() not in {".keras", ".h5"}:
            raise ValueError("Output model must use the .keras or .h5 format.")
        return name
    return f"{name}.keras"


def _output_path(value: str | Path, overwrite: bool) -> Path:
    output = Path(value).expanduser().resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}")
    if output.exists() and not output.is_file():
        raise FileExistsError(f"Output is not a file: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _validate_then_publish(output: Path, data: dict[str, Any], validator):
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".json",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        report, _ = validator(temporary)
        if not report.valid:
            messages = "; ".join(item["message"] for item in report.errors)
            raise ValueError(f"Generated definition is invalid: {messages}")
        os.replace(temporary, output)
        temporary = None
        return report
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _mel_cnn_definition(request: ModelGenerationRequest) -> dict[str, Any]:
    layers = [
        _layer("input", "input", 0, {"shape": request.input_shape, "dtype": "float32", "name": "mel_input"}),
        _layer("batch_norm", "batch_norm", 180),
        _layer("conv1", "conv2d", 360, _conv_parameters(16)),
        _layer("pool1", "max_pooling2d", 540, _pool_parameters()),
        _layer("conv2", "conv2d", 720, _conv_parameters(32)),
        _layer("pool2", "max_pooling2d", 900, _pool_parameters()),
        _layer("conv3", "conv2d", 1080, _conv_parameters(64)),
        _layer("pool3", "max_pooling2d", 1260, _pool_parameters()),
        _layer("global_pool", "global_avg_pooling2d", 1440),
        _layer("dense", "dense", 1620, {"units": 64, "activation": "relu", "use_bias": True, "kernel_initializer": "glorot_uniform"}),
        _layer("dropout", "dropout", 1800, {"rate": request.dropout}),
        _layer("classifier", "dense", 1980, {"units": request.classes, "activation": "softmax", "use_bias": True, "kernel_initializer": "glorot_uniform"}),
        _layer("output", "output", 2160, {"activation": "none", "optimizer": "Adam", "learning_rate": request.learning_rate, "loss": "sparse_categorical_crossentropy", "metrics": ["accuracy"]}),
    ]
    ids = [item["id"] for item in layers]
    return {
        "version": "1.0",
        "name": request.name,
        "description": "Generated Mel-CNN audio classification model.",
        "layers": layers,
        "connections": [
            {"source": source, "target": target}
            for source, target in zip(ids, ids[1:])
        ],
    }


def _layer(layer_id: str, layer_type: str, x: int, parameters: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "id": layer_id,
        "type": layer_type,
        "position": [x, 0],
        "parameters": parameters or {},
    }


def _conv_parameters(filters: int) -> dict[str, Any]:
    return {
        "filters": filters,
        "kernel_size": "(3, 3)",
        "strides": "(1, 1)",
        "padding": "same",
        "activation": "relu",
        "use_bias": True,
    }


def _pool_parameters() -> dict[str, Any]:
    return {"pool_size": "(2, 2)", "strides": "(2, 2)", "padding": "valid"}


def _classification_workflow(
    request: WorkflowGenerationRequest,
    *,
    dataset: Path,
    labels: Path,
    model: Path,
    output_model: str,
) -> dict[str, Any]:
    nodes = [
        _node("audio", "audio_folder", 0, 0, {"folder_path": str(dataset), "recursive": True, "auto_label": False, "target_sr": request.sample_rate, "max_files": 0, "selection_mode": "first_n"}),
        _node("labels", "label_file", 0, 180, {"file_path": str(labels), "format": labels.suffix.lower().lstrip("."), "filename_column": "filename", "label_column": "label"}),
        _node("mono", "channel_mapper", 180, 0, {"map1": "0", "map2": "", "map3": "", "map4": ""}),
        _node("align", "align_targets", 360, 0, {"match_mode": "basename", "missing_policy": "error", "duplicate_policy": "error", "fill_value": ""}),
        _node("split", "split", 540, 0, {"train_ratio": request.train_ratio, "val_ratio": request.val_ratio, "test_ratio": request.test_ratio, "shuffle": True, "stratify": True, "random_seed": request.seed}),
    ]
    for prefix, y in (("train", -180), ("val", 0), ("test", 180)):
        nodes.extend(
            [
                _node(f"{prefix}_trim", "trim_pad", 720, y, {"duration": request.duration, "mode": "pad_trim", "pad_mode": "constant", "position": "center"}),
                _node(f"{prefix}_norm", "normalize", 900, y, {"method": "peak", "target_level": -3.0}),
                _node(f"{prefix}_mel", "mel_spectrogram", 1080, y, {"n_mels": request.n_mels, "n_fft": request.n_fft, "hop_length": request.hop_length, "fmin": 0.0, "fmax": min(request.sample_rate / 2.0, 22050.0), "power_to_db": True}),
            ]
        )
    nodes.extend(
        [
            _node("model", "load_model", 1080, -360, {"model_path": str(model), "compile_model": True}),
            _node("trainer", "classification_trainer", 1260, 0, {"epochs": request.epochs, "batch_size": request.batch_size, "use_model_config": True, "optimizer": "Adam", "learning_rate": 0.001, "loss": "auto", "early_stopping": True, "patience": request.patience}),
            _node("save", "save_model", 1440, -90, {"save_mode": "new_file", "save_dir": "models", "file_name": output_model, "existing_file": "", "save_format": "keras", "confirm_overwrite": True}),
            _node("evaluate", "classification_evaluator", 1440, 90, {"batch_size": request.batch_size}),
        ]
    )
    connections = [
        _connection("audio", "audio", "mono", "audio"),
        _connection("mono", "out1", "align", "data"),
        _connection("audio", "file_paths", "align", "file_paths"),
        _connection("labels", "label_map", "align", "target_map"),
        _connection("align", "aligned_data", "split", "data"),
        _connection("align", "aligned_targets", "split", "targets"),
        _connection("labels", "label_map", "split", "target_metadata"),
    ]
    for prefix, split_port in (("train", "train"), ("val", "val"), ("test", "test")):
        connections.extend(
            [
                _connection("split", f"{split_port}_data", f"{prefix}_trim", "audio"),
                _connection(f"{prefix}_trim", "audio", f"{prefix}_norm", "audio"),
                _connection(f"{prefix}_norm", "audio", f"{prefix}_mel", "audio"),
            ]
        )
    connections.extend(
        [
            _connection("model", "model", "trainer", "model"),
            _connection("train_mel", "feature", "trainer", "x_train"),
            _connection("split", "train_targets", "trainer", "y_train"),
            _connection("split", "target_metadata", "trainer", "target_metadata"),
            _connection("val_mel", "feature", "trainer", "x_val"),
            _connection("split", "val_targets", "trainer", "y_val"),
            _connection("trainer", "trained_model", "save", "model"),
            _connection("trainer", "trained_model", "evaluate", "model"),
            _connection("test_mel", "feature", "evaluate", "x_test"),
            _connection("split", "test_targets", "evaluate", "y_test"),
            _connection("split", "target_metadata", "evaluate", "target_metadata"),
        ]
    )
    return {
        "version": "1.0",
        "metadata": {
            "name": request.name,
            "description": "Generated audio classification training workflow.",
            "author": "audio-platform-cli",
        },
        "nodes": nodes,
        "connections": connections,
    }


def _node(node_id: str, node_type: str, x: int, y: int, parameters: dict[str, Any]) -> dict[str, Any]:
    return {"id": node_id, "type": node_type, "position": [x, y], "parameters": parameters}


def _connection(source_node: str, source_port: str, target_node: str, target_port: str) -> dict[str, Any]:
    return {
        "source": {"node": source_node, "port": source_port},
        "target": {"node": target_node, "port": target_port},
    }
