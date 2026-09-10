"""Shared anomaly persistence and complete export; joblib is trusted code only."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path, PureWindowsPath
import shutil
import sys
import tempfile
import warnings
import zipfile

import numpy as np

from src.ui.i18n import tr_


FORMAT_VERSION = 1
MAX_METADATA_BYTES = 1024 * 1024
MAX_PAYLOAD_BYTES = 1024 * 1024 * 1024


def runtime_versions():
    """Return the serialization compatibility fingerprint."""
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        **{name: importlib.metadata.version(name)
           for name in ("numpy", "scipy", "scikit-learn", "joblib")},
    }


def sha256_file(path):
    """Hash a file without reading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_target(folder, name):
    """Accept a directory and one portable basename, never a nested path."""
    if not isinstance(folder, str) or not folder.strip():
        raise ValueError(tr_("Output directory is required"))
    if (not isinstance(name, str) or not name or name != name.strip()
            or name in (".", "..") or name.endswith(".")
            or any(ord(char) < 32 or char in '<>:"/\\|?*' for char in name)
            or PureWindowsPath(name).is_reserved()):
        raise ValueError(tr_("Output name must be a valid single filename"))
    return Path(folder).expanduser().resolve() / name


def _new_target(path):
    path = Path(path).expanduser().absolute()
    if os.path.lexists(path):
        raise FileExistsError(tr_("Output already exists: {path}").format(path=path))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8")


def _vector(value, name, count=None):
    array = np.asarray(value, dtype=np.float64)
    if (array.ndim != 1 or not np.all(np.isfinite(array))
            or (count is not None and len(array) != count)):
        raise ValueError(tr_("Invalid or misaligned anomaly array: {name}").format(name=name))
    return array


def validate_model(artifact):
    """Validate supported scoring state after trust has been established."""
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import RobustScaler, StandardScaler
    from sklearn.utils.validation import check_is_fitted
    from .nodes.anomaly import AnomalyModelArtifact

    if not isinstance(artifact, AnomalyModelArtifact):
        raise TypeError(tr_("File does not contain an anomaly model artifact"))
    if (artifact.algorithm != "isolation_forest" or type(artifact.estimator) is not IsolationForest
            or type(artifact.feature_count) not in (int, np.int64, np.int32)
            or artifact.feature_count < 1):
        raise ValueError(tr_("Invalid anomaly model scoring state"))
    check_is_fitted(artifact.estimator)
    if artifact.estimator.n_features_in_ != artifact.feature_count:
        raise ValueError(tr_("Invalid anomaly model scoring state"))
    if artifact.scaler is not None:
        if type(artifact.scaler) not in (StandardScaler, RobustScaler):
            raise ValueError(tr_("Invalid anomaly model scoring state"))
        check_is_fitted(artifact.scaler)
        if artifact.scaler.n_features_in_ != artifact.feature_count:
            raise ValueError(tr_("Invalid anomaly model scoring state"))
        for name in ("mean_", "center_", "scale_", "var_"):
            value = getattr(artifact.scaler, name, None)
            if value is not None:
                _vector(value, name, artifact.feature_count)
    bounds = _vector(artifact.score_bounds, "score_bounds", 2)
    reference = _vector(artifact.reference_scores, "reference_scores")
    if bounds[0] > bounds[1] or len(reference) < 2:
        raise ValueError(tr_("Invalid anomaly model scoring state"))
    if not isinstance(artifact.feature_schema, dict) or not isinstance(artifact.training_summary, dict):
        raise ValueError(tr_("Invalid anomaly model scoring state"))
    if artifact.feature_schema.get("feature_count", artifact.feature_count) != artifact.feature_count:
        raise ValueError(tr_("Invalid anomaly model scoring state"))
    _json_bytes(artifact.feature_schema)
    _json_bytes(artifact.training_summary)


def _model_metadata(artifact):
    return {
        "format": "dt-anomaly-model", "format_version": FORMAT_VERSION,
        "runtime": runtime_versions(), "algorithm": artifact.algorithm,
        "feature_count": int(artifact.feature_count),
        "feature_schema": artifact.feature_schema,
        "score_bounds": list(map(float, artifact.score_bounds)),
        "reference_count": len(artifact.reference_scores),
        "training_summary": artifact.training_summary,
        "score_direction": "higher_is_more_anomalous",
        "decision_policy": "stored_in_workflow",
    }


def save_model(artifact, path):
    """Publish a complete ZIP using an atomic, exclusive hard link."""
    import joblib

    validate_model(artifact)
    target = _new_target(path)
    descriptor, staging = tempfile.mkstemp(prefix=".anomaly-", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w+b") as stream:
            payload = io.BytesIO()
            joblib.dump(artifact, payload)
            data = payload.getvalue()
            if len(data) > MAX_PAYLOAD_BYTES:
                raise ValueError(tr_("Anomaly model exceeds the supported size limit"))
            metadata = _model_metadata(artifact)
            metadata["payload_sha256"] = hashlib.sha256(data).hexdigest()
            metadata_bytes = _json_bytes(metadata)
            if len(metadata_bytes) > MAX_METADATA_BYTES:
                raise ValueError(tr_("Anomaly model exceeds the supported size limit"))
            with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
                archive.writestr("metadata.json", metadata_bytes)
                archive.writestr("model.joblib", data)
            stream.flush()
            os.fsync(stream.fileno())
        # link fails if any target already exists; os.replace would overwrite it.
        os.link(staging, target)
    finally:
        Path(staging).unlink(missing_ok=True)
    return str(target)


def _read_bundle(path):
    """Inspect JSON and bytes only. Never import or call joblib here."""
    with zipfile.ZipFile(path) as archive:
        if sorted(archive.namelist()) != ["metadata.json", "model.joblib"]:
            raise ValueError(tr_("Invalid anomaly model archive"))
        if (archive.getinfo("metadata.json").file_size > MAX_METADATA_BYTES
                or archive.getinfo("model.joblib").file_size > MAX_PAYLOAD_BYTES):
            raise ValueError(tr_("Anomaly model exceeds the supported size limit"))
        metadata = json.loads(archive.read("metadata.json"))
        if (not isinstance(metadata, dict) or metadata.get("format") != "dt-anomaly-model"
                or type(metadata.get("format_version")) is not int
                or metadata["format_version"] != FORMAT_VERSION):
            raise ValueError(tr_("Unsupported anomaly model format version"))
        if metadata.get("runtime") != runtime_versions():
            raise ValueError(tr_("Anomaly model runtime versions do not match this environment"))
        if (metadata.get("algorithm") != "isolation_forest"
                or type(metadata.get("feature_count")) is not int or metadata["feature_count"] < 1
                or type(metadata.get("reference_count")) is not int or metadata["reference_count"] < 2
                or not isinstance(metadata.get("feature_schema"), dict)
                or not isinstance(metadata.get("training_summary"), dict)
                or metadata.get("score_direction") != "higher_is_more_anomalous"
                or metadata.get("decision_policy") != "stored_in_workflow"):
            raise ValueError(tr_("Invalid anomaly model scoring state"))
        bounds = _vector(metadata.get("score_bounds"), "score_bounds", 2)
        if bounds[0] > bounds[1]:
            raise ValueError(tr_("Invalid anomaly model scoring state"))
        _json_bytes(metadata)
        payload = archive.read("model.joblib")
        if hashlib.sha256(payload).hexdigest() != metadata.get("payload_sha256"):
            raise ValueError(tr_("Anomaly model payload checksum mismatch"))
    return metadata, payload


def inspect_model(path):
    """Return untrusted metadata without deserializing executable model state."""
    metadata, _ = _read_bundle(path)
    return metadata


def load_model(path, *, trusted=False, format="anomaly_zip"):
    """Deserialize only with explicit authorization; checks are not a sandbox."""
    if trusted is not True:
        raise PermissionError(tr_("Loading joblib can execute code; explicitly trust the model before loading"))
    import joblib

    if format == "anomaly_zip":
        metadata, payload = _read_bundle(path)
        artifact = joblib.load(io.BytesIO(payload))
    elif format == "legacy_joblib":
        warnings.warn(tr_("Legacy joblib model: runtime version compatibility cannot be verified"),
                      UserWarning, stacklevel=2)
        metadata = None
        artifact = joblib.load(path)
    else:
        raise ValueError(tr_("Unsupported anomaly model format"))
    validate_model(artifact)
    if metadata is not None:
        expected = _model_metadata(artifact)
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError(tr_("Anomaly model metadata does not match scoring state"))
    return artifact


def export_results(data, path, *, spreadsheet_safe=False):
    """Export all aligned rows into a newly published directory."""
    from .nodes.anomaly import AnomalyResultData, AnomalyScoresData

    result = data if isinstance(data, AnomalyResultData) else None
    scores = result.scores if result is not None else data
    if not isinstance(scores, AnomalyScoresData):
        raise TypeError(tr_("Export requires anomaly scores or an anomaly result"))
    raw = _vector(scores.raw_scores, "raw_scores")
    count = len(raw)
    normalized = _vector(scores.normalized_scores, "normalized_scores", count)
    _vector(scores.reference_normalized_scores, "reference_normalized_scores")
    if (len(scores.sample_ids) != count or not all(isinstance(s, str) for s in scores.sample_ids)
            or (scores.source_items and len(scores.source_items) != count)):
        raise ValueError(tr_("Score arrays and sample IDs must have the same length"))
    if np.any((normalized < 0) | (normalized > 100)):
        raise ValueError(tr_("Normalized anomaly scores must be between 0 and 100"))
    metadata = {
        "format": "dt-anomaly-results", "format_version": FORMAT_VERSION,
        "kind": "decisions" if result is not None else "scores", "sample_count": count,
        "score_direction": "higher_is_more_anomalous", "row_order": "input",
        "sample_id_encoding": "apostrophe_prefix_all" if spreadsheet_safe else "verbatim",
        "score_metadata": scores.metadata,
    }
    fields = ["row_index", "sample_id", "raw_score", "normalized_score"]
    if result is not None:
        decisions = np.asarray(result.is_anomaly)
        if (decisions.ndim != 1 or decisions.dtype != np.bool_ or len(decisions) != count
                or len(result.severities) != count
                or any(s not in ("normal", "attention", "anomaly") for s in result.severities)
                or not np.isfinite([result.threshold, result.attention_threshold]).all()
                or not 0 <= result.attention_threshold <= result.threshold <= 100
                or result.strategy not in ("reference_quantile", "mean_std", "median_mad", "manual")):
            raise ValueError(tr_("Invalid or misaligned anomaly decisions"))
        expected = ["anomaly" if x >= result.threshold else "attention"
                    if x >= result.attention_threshold else "normal" for x in normalized]
        if (not np.array_equal(decisions, normalized >= result.threshold)
                or list(result.severities) != expected):
            raise ValueError(tr_("Invalid or misaligned anomaly decisions"))
        fields += ["is_anomaly", "severity"]
        metadata.update(threshold=float(result.threshold),
                        attention_threshold=float(result.attention_threshold), strategy=result.strategy)
    metadata["columns"] = fields
    _json_bytes(metadata)
    target = _new_target(path)
    staging = Path(tempfile.mkdtemp(prefix=".anomaly-export-", dir=target.parent))
    try:
        csv_path = staging / "results.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(fields)
            for index in range(count):
                sample_id = scores.sample_ids[index]
                row = [index, "'" + sample_id if spreadsheet_safe else sample_id,
                       float(raw[index]), float(normalized[index])]
                if result is not None:
                    row += [int(result.is_anomaly[index]), result.severities[index]]
                writer.writerow(row)
            stream.flush()
            os.fsync(stream.fileno())
        metadata["csv_sha256"] = sha256_file(csv_path)
        with (staging / "metadata.json").open("wb") as stream:
            stream.write(_json_bytes(metadata))
            stream.flush()
            os.fsync(stream.fileno())
        if os.path.lexists(target):
            raise FileExistsError(tr_("Output already exists: {path}").format(path=target))
        # Windows rename refuses an existing destination, including empty folders.
        os.rename(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {"directory": str(target), "sample_count": count,
            "csv_path": str(target / "results.csv"),
            "metadata_path": str(target / "metadata.json"),
            "csv_sha256": metadata["csv_sha256"],
            "metadata_sha256": sha256_file(target / "metadata.json")}
