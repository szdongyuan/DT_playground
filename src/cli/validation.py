"""Side-effect-free validation for workflow and model definitions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.model_builder import ModelGraph, get_layer_class
from src.workflow import Workflow
from src.workflow.node_base import get_node_class


INPUT_PATH_PARAMETERS = {
    "audio_folder": ("folder_path",),
    "sqlite_audio_database": ("database_path", "audio_root"),
    "audio_file": ("file_path",),
    "label_file": ("file_path",),
    "target_file": ("file_path",),
    "load_model": ("model_path",),
}


@dataclass
class ValidationReport:
    """Structured validation result."""

    valid: bool = True
    errors: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    data_checks: dict[str, Any] | None = None

    def error(self, code: str, message: str, **context: Any) -> None:
        self.valid = False
        self.errors.append({"code": code, "message": message, "context": context})

    def warning(self, code: str, message: str, **context: Any) -> None:
        self.warnings.append({"code": code, "message": message, "context": context})

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema_version": "1.0",
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
        }
        if self.data_checks is not None:
            result["data_checks"] = self.data_checks
        return result


def validate_workflow_file(path_value: str | Path, *, headless: bool = True, check_data: bool = False) -> tuple[ValidationReport, Workflow | None]:
    """Validate raw workflow JSON before creating runtime objects."""
    path = Path(path_value).expanduser().resolve()
    report = ValidationReport()
    if check_data:
        report.data_checks = {"status": "skipped", "reason": "Structural validation failed."}
    data = _read_json(path, report, "workflow")
    if data is None:
        return report, None
    if not isinstance(data, dict):
        report.error("workflow.invalid_root", "Workflow JSON root must be an object.")
        return report, None

    nodes = data.get("nodes")
    connections = data.get("connections")
    if not isinstance(nodes, list):
        report.error("workflow.nodes_type", "Workflow 'nodes' must be an array.")
        return report, None
    if not isinstance(connections, list):
        report.error("workflow.connections_type", "Workflow 'connections' must be an array.")
        return report, None

    seen_ids: set[str] = set()
    node_objects: dict[str, Any] = {}
    for index, node_data in enumerate(nodes):
        if not isinstance(node_data, dict):
            report.error("node.invalid", "Node must be an object.", index=index)
            continue
        node_id = str(node_data.get("id", ""))
        node_type = str(node_data.get("type", ""))
        if not node_id:
            report.error("node.missing_id", "Node ID is required.", index=index)
        elif node_id in seen_ids:
            report.error("node.duplicate_id", f"Duplicate node ID: {node_id}", node_id=node_id)
        seen_ids.add(node_id)
        node_class = get_node_class(node_type)
        if node_class is None:
            report.error("node.unknown_type", f"Unknown node type: {node_type}", node_id=node_id)
            continue
        node = node_class(node_id=node_id or None)
        if node_id:
            node_objects[node_id] = node
        parameters = node_data.get("parameters", {})
        if not isinstance(parameters, dict):
            report.error("node.parameters_type", "Node parameters must be an object.", node_id=node_id)
            continue
        for name, value in parameters.items():
            if name not in node.parameters:
                report.error("node.unknown_parameter", f"Unknown parameter '{name}'.", node_id=node_id)
                continue
            ok, message = node.set_parameter(name, value)
            if not ok:
                report.error("node.invalid_parameter", message, node_id=node_id, parameter=name)
        if headless and node_type == "show_history":
            report.error("headless.interactive_node", "Show training history is GUI-dependent; save or export metrics instead.", node_id=node_id)
        if headless and getattr(node, "is_breakpoint", False) and node.get_parameter("enabled") is not False:
            report.error("headless.breakpoint", "Enabled breakpoints are not allowed in headless execution.", node_id=node_id)
        _validate_input_paths(node, path.parent, report)

    _validate_connections(connections, node_objects, report)

    if report.valid:
        try:
            workflow = Workflow.from_dict(data)
            valid, errors = workflow.validate()
            if not valid:
                for message in errors:
                    report.error("workflow.validation", message)
        except Exception as exc:
            report.error("workflow.load", f"Unable to load workflow: {exc}")
            workflow = None
    else:
        workflow = None
    if check_data:
        if report.valid:
            from src.cli.data_validation import check_workflow_data

            check_workflow_data(node_objects, connections, path.parent, report)
        else:
            report.data_checks = {"status": "skipped", "reason": "Structural validation failed."}
    return report, workflow


def validate_model_file(path_value: str | Path, *, build: bool = False) -> tuple[ValidationReport, ModelGraph | None]:
    """Validate a model definition and optionally build it to verify shapes."""
    path = Path(path_value).expanduser().resolve()
    report = ValidationReport()
    data = _read_json(path, report, "model")
    if data is None:
        return report, None
    if not isinstance(data, dict):
        report.error("model.invalid_root", "Model JSON root must be an object.")
        return report, None
    layers = data.get("layers")
    if not isinstance(layers, list):
        report.error("model.layers_type", "Model 'layers' must be an array.")
        return report, None

    seen_ids: set[str] = set()
    for index, layer_data in enumerate(layers):
        if not isinstance(layer_data, dict):
            report.error("layer.invalid", "Layer must be an object.", index=index)
            continue
        layer_id = str(layer_data.get("id", ""))
        layer_type = str(layer_data.get("type", ""))
        if not layer_id:
            report.error("layer.missing_id", "Layer ID is required.", index=index)
        elif layer_id in seen_ids:
            report.error("layer.duplicate_id", f"Duplicate layer ID: {layer_id}", layer_id=layer_id)
        seen_ids.add(layer_id)
        layer_class = get_layer_class(layer_type)
        if layer_class is None:
            report.error("layer.unknown_type", f"Unknown layer type: {layer_type}", layer_id=layer_id)
            continue
        layer = layer_class(layer_id=layer_id or None)
        parameters = layer_data.get("parameters", {})
        if not isinstance(parameters, dict):
            report.error("layer.parameters_type", "Layer parameters must be an object.", layer_id=layer_id)
            continue
        for name, value in parameters.items():
            if name not in layer.parameters:
                report.error("layer.unknown_parameter", f"Unknown parameter '{name}'.", layer_id=layer_id)
                continue
            ok, message = layer.set_parameter(name, value)
            if not ok:
                report.error("layer.invalid_parameter", message, layer_id=layer_id, parameter=name)

    graph = None
    if report.valid:
        try:
            graph = ModelGraph.from_dict(data)
            valid, errors = graph.validate()
            if not valid:
                for message in errors:
                    report.error("model.validation", message)
            if build and report.valid:
                graph.build_keras_model(compile_model=False)
        except Exception as exc:
            report.error("model.build" if build else "model.load", str(exc))
    return report, graph


def _read_json(path: Path, report: ValidationReport, kind: str) -> Any | None:
    if not path.is_file():
        report.error(f"{kind}.not_found", f"{kind.title()} file does not exist: {path}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        report.error(f"{kind}.invalid_json", f"Unable to read {kind} JSON: {exc}")
        return None


def _validate_input_paths(node, base_dir: Path, report: ValidationReport) -> None:
    for name in INPUT_PATH_PARAMETERS.get(node.node_type, ()):
        value = node.get_parameter(name)
        if not value:
            continue
        path = Path(value).expanduser()
        resolved = path if path.is_absolute() else (base_dir / path)
        if not resolved.exists():
            report.error(
                "path.not_found",
                f"Input path does not exist: {resolved.resolve()}",
                node_id=node.node_id,
                parameter=name,
            )


def _validate_connections(
    connections: list[Any],
    nodes: dict[str, Any],
    report: ValidationReport,
) -> None:
    occupied_inputs: set[tuple[str, str]] = set()
    seen_connections: set[tuple[str, str, str, str]] = set()
    for index, data in enumerate(connections):
        if not isinstance(data, dict):
            report.error("connection.invalid", "Connection must be an object.", index=index)
            continue
        source = data.get("source", {})
        target = data.get("target", {})
        if not isinstance(source, dict) or not isinstance(target, dict):
            report.error("connection.invalid", "Connection endpoints must be objects.", index=index)
            continue
        source_id = str(source.get("node", ""))
        source_port = str(source.get("port", ""))
        target_id = str(target.get("node", ""))
        target_port = str(target.get("port", ""))
        key = (source_id, source_port, target_id, target_port)
        if key in seen_connections:
            report.error("connection.duplicate", "Duplicate connection.", index=index)
            continue
        seen_connections.add(key)
        source_node = nodes.get(source_id)
        target_node = nodes.get(target_id)
        if source_node is None or target_node is None:
            report.error(
                "connection.unknown_node",
                "Connection references an unknown node.",
                index=index,
                source_node=source_id,
                target_node=target_id,
            )
            continue
        if source_port not in source_node.outputs or target_port not in target_node.inputs:
            report.error(
                "connection.unknown_port",
                "Connection references an unknown port.",
                index=index,
                source_port=source_port,
                target_port=target_port,
            )
            continue
        if not source_node.outputs[source_port].can_connect_to(target_node.inputs[target_port]):
            report.error("connection.incompatible", "Connection port types are incompatible.", index=index)
        input_key = (target_id, target_port)
        if input_key in occupied_inputs and not target_node.inputs[target_port].multi_connection:
            report.error("connection.input_occupied", "Input port has multiple connections.", index=index)
        occupied_inputs.add(input_key)
