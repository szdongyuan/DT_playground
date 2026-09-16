"""Shared workflow input-path resolution for GUI and CLI execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any


INPUT_PATH_PARAMETERS = {
    "audio_folder": ("folder_path",),
    "sqlite_audio_database": ("database_path", "audio_root"),
    "audio_file": ("file_path",),
    "label_file": ("file_path",),
    "target_file": ("file_path",),
    "align_targets": ("dataset_root",),
    "load_model": ("model_path",),
    "load_anomaly_model": ("model_path",),
}


def workflow_definition_dir(workflow) -> Path | None:
    """Return the directory containing a loaded workflow, when known."""
    source = getattr(workflow, "_file_path", None)
    if not source:
        return None
    return Path(source).expanduser().resolve().parent


def resolve_workflow_input_paths(
    workflow,
    definition_dir: str | Path | None = None,
) -> dict[tuple[str, str], Any]:
    """Resolve relative input parameters and return values needed for restoration."""
    base = (
        Path(definition_dir).expanduser().resolve()
        if definition_dir is not None
        else workflow_definition_dir(workflow)
    )
    if base is None:
        return {}

    originals: dict[tuple[str, str], Any] = {}
    for node in workflow.nodes.values():
        for name in INPUT_PATH_PARAMETERS.get(node.node_type, ()):
            value = node.get_parameter(name)
            if not value:
                continue
            path = Path(value).expanduser()
            if path.is_absolute():
                continue
            originals[(node.node_id, name)] = value
            node.parameter_values[name] = str((base / path).resolve())
    return originals


def restore_workflow_input_paths(
    workflow,
    originals: dict[tuple[str, str], Any],
) -> None:
    """Restore serialized relative values after an execution finishes."""
    for (node_id, name), value in originals.items():
        node = workflow.get_node(node_id)
        if node is not None:
            node.parameter_values[name] = value
