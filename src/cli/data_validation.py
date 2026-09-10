"""Explicit read-only preflight; never dispatch the workflow execution engine."""

from collections import Counter
from pathlib import Path
import warnings

import numpy as np
import soundfile as sf

from src.workflow.nodes.data_source import AudioFolderNode


def _resolve(value, base_dir):
    if not value:
        raise ValueError("An input path is required.")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base_dir / path).resolve()


def _decode_info(path):
    """Decode one file in bounded blocks, with the runtime's format fallback."""
    try:
        handle = sf.SoundFile(str(path))
    except Exception:
        import librosa

        data, rate = librosa.load(str(path), sr=None, mono=False)
        if not data.size or not np.isfinite(data).all():
            raise ValueError("Audio is empty or contains non-finite samples.")
        return (1 if data.ndim == 1 else data.shape[0], data.shape[-1], rate)
    with handle:
        count = 0
        for block in handle.blocks(blocksize=65536, dtype="float32", always_2d=True):
            if not np.isfinite(block).all():
                raise ValueError("Audio contains non-finite samples.")
            count += len(block)
        if count == 0:
            raise ValueError("Audio contains no samples.")
        if count != handle.frames:
            raise ValueError("Decoded frame count differs from audio metadata.")
        return handle.channels, count, handle.samplerate


def _check_audio(node, base_dir, report):
    folder = node.node_type == "audio_folder"
    root = _resolve(node.get_parameter("folder_path" if folder else "file_path"), base_dir)
    if folder:
        if not root.is_dir():
            raise ValueError(f"Not an audio directory: {root}")
        candidates = root.rglob("*") if node.get_parameter("recursive") else root.glob("*")
        files = sorted(p for p in candidates if p.is_file() and p.suffix.lower() in AudioFolderNode.SUPPORTED_FORMATS)
    else:
        if not root.is_file():
            raise ValueError(f"Not an audio file: {root}")
        files = [root]
    candidate_count = len(files)
    if folder and node.get_parameter("max_files"):
        if node.get_parameter("selection_mode") == "random_n":
            report.warning("data.random_selection", "All candidates are checked; the runtime random subset is not predicted.", node_id=node.node_id)
        else:
            files = files[:node.get_parameter("max_files")]

    readable, shapes, rates = [], Counter(), Counter()
    for path in files:
        try:
            with warnings.catch_warnings(record=True) as decoder_warnings:
                warnings.simplefilter("always")
                channels, samples, rate = _decode_info(path)
            shapes[(int(channels), int(samples))] += 1
            rates[int(rate)] += 1
            readable.append(str(path))
        except Exception as exc:
            report.error("data.unreadable_audio", str(exc), node_id=node.node_id, path=str(path))
        finally:
            for warning in decoder_warnings:
                report.warning("data.decoder_warning", str(warning.message), node_id=node.node_id, path=str(path))
    if not readable:
        report.error("data.empty_audio", "No readable non-empty audio files were found.", node_id=node.node_id, path=str(root))
    if len(shapes) > 1:
        report.warning("data.mixed_shapes", "Raw audio shapes differ; downstream preprocessing must handle this.", node_id=node.node_id)
    report.data_checks["sources"].append({
        "node_id": node.node_id, "path": str(root), "candidate_files": candidate_count,
        "checked_files": len(files), "valid_files": len(readable),
        "invalid_files": len(files) - len(readable),
        "raw_shapes": [{"shape": list(shape), "count": count} for shape, count in sorted(shapes.items())],
        "sample_rates": dict(sorted(rates.items())),
    })
    if not folder:
        return {}
    names, labels = {}, []
    for path in readable:
        relative = Path(path).relative_to(root)
        name = relative.parts[0] if len(relative.parts) > 1 else "default"
        if not node.get_parameter("auto_label"):
            name = "default"
        if name not in names:
            names[name] = len(names)
        labels.append(names[name])
    return {"file_paths": readable, "labels": labels,
            "label_map": {"label_names": names, "filename_map": dict(zip(readable, labels))}}


def check_workflow_data(nodes, connections, base_dir, report):
    """Check sources and direct alignment using the existing nodes' contracts.

    Only the known read-only label readers and string-only alignment node may
    execute. Audio, preprocessing, training, model loading and sinks never do.
    """
    report.data_checks = {
        "status": "completed", "sources": [], "alignments": [],
        "model_input_compatibility": "not_checked",
    }
    report.warning("data.shape_scope", "Only raw audio shapes are inspected; transformed features and model input compatibility are not checked.")
    outputs = {}
    for node_id, node in nodes.items():
        try:
            if node.node_type in {"audio_folder", "audio_file"}:
                outputs[node_id] = _check_audio(node, base_dir, report)
            elif node.node_type in {"label_file", "target_file"}:
                node.set_parameter("file_path", str(_resolve(node.get_parameter("file_path"), base_dir)))
                if not node.execute():
                    raise ValueError(node.error_message)
                outputs[node_id] = {name: port.data for name, port in node.outputs.items()}
                values = outputs[node_id].get("labels" if node.node_type == "label_file" else "targets")
                if values is None or len(values) == 0:
                    raise ValueError("Target file contains no targets.")
            elif not node.inputs:
                report.warning("data.unsupported_source", "This source type is not checked by data preflight.", node_id=node_id, node_type=node.node_type)
        except Exception as exc:
            report.error("data.source", str(exc), node_id=node_id)

    incoming = {(c["target"]["node"], c["target"]["port"]): c["source"] for c in connections}
    for node_id, node in nodes.items():
        if node.node_type != "align_targets":
            continue
        available = {}
        unresolved = []
        for port in ("file_paths", "target_map", "targets"):
            source = incoming.get((node_id, port))
            if source is None:
                continue
            source_outputs = outputs.get(source["node"], {})
            if source["port"] in source_outputs:
                available[port] = source_outputs[source["port"]]
            else:
                unresolved.append(port)
        if unresolved or "file_paths" not in available or not ({"target_map", "targets"} & available.keys()):
            report.warning("data.alignment_skipped", "Alignment requires direct checked sources; an upstream transformation or failed source prevents checking.", node_id=node_id)
            report.data_checks["alignments"].append({"node_id": node_id, "status": "skipped"})
            continue
        for port, value in available.items():
            node.inputs[port].data = value
        if not node.execute():
            report.error("data.alignment", node.error_message, node_id=node_id)
            report.data_checks["alignments"].append({"node_id": node_id, "status": "failed"})
            continue
        alignment = node.outputs["alignment_report"].data
        report.data_checks["alignments"].append({"node_id": node_id, "status": "completed", **alignment})
        if not node.outputs["aligned_file_paths"].data:
            report.error("data.empty_alignment", "No samples remain after target alignment.", node_id=node_id)
        elif alignment["missing_count"] or alignment["duplicate_count"] or alignment["unused_count"]:
            report.warning("data.alignment_policy", "Alignment used drop/fill/duplicate policies or left unused targets; inspect the alignment report.", node_id=node_id)
    if not report.valid:
        report.data_checks["status"] = "failed"
