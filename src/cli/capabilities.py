"""Machine-readable discovery of workflow nodes and model layers."""

from __future__ import annotations

from typing import Any

from src.cli.contracts import CLI_SCHEMA_VERSION, json_safe
from src.model_builder import get_all_layer_types, get_layer_class
from src.workflow import nodes as _workflow_nodes  # noqa: F401
from src.workflow.node_base import get_all_node_types, get_node_class


def _parameter_schema(parameter) -> dict[str, Any]:
    return json_safe(parameter.to_dict())


def _port_schema(port) -> dict[str, Any]:
    return {
        "name": port.name,
        "display_name": port.display_name,
        "data_type": port.data_type.value,
        "required": port.required,
        "multi_connection": port.multi_connection,
        "default_value": json_safe(port.default_value),
        "description": port.description,
    }


def get_capabilities(include_hidden: bool = False) -> dict[str, Any]:
    """Return a deterministic capability catalog for automation clients."""
    node_items = []
    for node_type in sorted(get_all_node_types()):
        node_class = get_node_class(node_type)
        if node_class is None:
            continue
        if not include_hidden and not getattr(node_class, "visible_in_palette", True):
            continue
        node = node_class()
        node_items.append(
            {
                "type": node_type,
                "display_name": node.display_name,
                "description": node.description,
                "category": node.category.value,
                "subcategory": getattr(node, "subcategory", ""),
                "headless": not _is_interactive_node(node),
                "inputs": [_port_schema(port) for port in node.inputs.values()],
                "outputs": [_port_schema(port) for port in node.outputs.values()],
                "parameters": [
                    _parameter_schema(parameter)
                    for parameter in node.parameters.values()
                ],
            }
        )

    layer_items = []
    for layer_type in sorted(get_all_layer_types()):
        layer_class = get_layer_class(layer_type)
        if layer_class is None:
            continue
        layer = layer_class()
        layer_items.append(
            {
                "type": layer_type,
                "display_name": layer.display_name,
                "description": layer.description,
                "category": layer.category.value,
                "keras_class": layer.keras_class,
                "parameters": [
                    _parameter_schema(parameter)
                    for parameter in layer.parameters.values()
                ],
            }
        )

    return {
        "schema_version": CLI_SCHEMA_VERSION,
        "scope": "audio_acoustic",
        "workflow_version": "1.0",
        "model_version": "1.0",
        "nodes": node_items,
        "layers": layer_items,
        "commands": ["capabilities", "inspect", "validate", "build-model", "run"],
    }


def _is_interactive_node(node) -> bool:
    if node.node_type == "show_history":
        return True
    return bool(
        getattr(node, "is_breakpoint", False)
        and node.get_parameter("enabled") is not False
    )
