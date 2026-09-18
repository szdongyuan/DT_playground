"""Version two bundles separate neural networks from portable scoring metadata."""

from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import zipfile

from src.ui.i18n import tr_
from . import anomaly_io as legacy


def save_bundle(artifact, path):
    import joblib
    target = legacy._new_target(path)
    with tempfile.TemporaryDirectory(prefix=".anomaly-v2-", dir=target.parent) as directory:
        root = Path(directory)
        state = replace(artifact, estimator=None) if artifact.algorithm == "autoencoder" else artifact
        state_bytes = io.BytesIO()
        joblib.dump(state, state_bytes)
        payloads = {"model.joblib": state_bytes.getvalue()}
        if artifact.algorithm == "autoencoder":
            artifact.estimator.save(root / "network.keras")
            payloads["network.keras"] = (root / "network.keras").read_bytes()
        state_json = {"preprocessing": artifact.preprocessing_state.to_dict() if artifact.preprocessing_state else None,
                      "reference_parent_ids": artifact.reference_parent_ids,
                      "reference_sample_ids": artifact.reference_sample_ids}
        payloads["state.json"] = legacy._json_bytes(state_json)
        metadata = legacy._model_metadata(artifact)
        metadata["format_version"] = 2
        if artifact.algorithm == "autoencoder":
            import importlib.metadata
            metadata["neural_runtime"] = {name: importlib.metadata.version(name) for name in ("tensorflow", "keras")}
        metadata["payloads"] = {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}
        encoded = legacy._json_bytes(metadata)
        if len(encoded) > legacy.MAX_METADATA_BYTES or any(len(data) > legacy.MAX_PAYLOAD_BYTES for data in payloads.values()):
            raise ValueError(tr_("Anomaly model exceeds the supported size limit"))
        staging = root / "bundle.tmp"
        with staging.open("w+b") as stream:
            with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
                archive.writestr("metadata.json", encoded)
                for name, data in payloads.items():
                    archive.writestr(name, data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(staging, target)
    return str(target)


def read_bundle(path):
    """Validate names, sizes, versions and hashes before any deserialization."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if (len(names) != len(set(names)) or "metadata.json" not in names
                or archive.getinfo("metadata.json").file_size > legacy.MAX_METADATA_BYTES):
            raise ValueError(tr_("Invalid anomaly model archive"))
        metadata = json.loads(archive.read("metadata.json"))
        if (not isinstance(metadata, dict) or type(metadata.get("format_version")) is not int
                or metadata["format_version"] != 2 or metadata.get("format") != "dt-anomaly-model"
                or metadata.get("algorithm") not in ("isolation_forest", "knn", "autoencoder")):
            raise ValueError(tr_("Unsupported anomaly model format version"))
        expected = {"metadata.json", "state.json", "model.joblib"}
        if metadata["algorithm"] == "autoencoder":
            import importlib.metadata
            expected.add("network.keras")
            if metadata.get("neural_runtime") != {name: importlib.metadata.version(name) for name in ("tensorflow", "keras")}:
                raise ValueError(tr_("Neural model runtime versions do not match this environment"))
        if set(names) != expected or set(metadata.get("payloads", {})) != expected - {"metadata.json"}:
            raise ValueError(tr_("Invalid anomaly model archive"))
        if metadata.get("runtime") != legacy.runtime_versions():
            raise ValueError(tr_("Anomaly model runtime versions do not match this environment"))
        if (type(metadata.get("feature_count")) is not int or metadata["feature_count"] < 1
                or type(metadata.get("reference_count")) is not int or metadata["reference_count"] < 2
                or not isinstance(metadata.get("feature_schema"), dict)
                or not isinstance(metadata.get("training_summary"), dict)
                or metadata.get("score_direction") != "higher_is_more_anomalous"
                or metadata.get("decision_policy") != "stored_in_workflow"):
            raise ValueError(tr_("Invalid anomaly model scoring state"))
        bounds = legacy._vector(metadata.get("score_bounds"), "score_bounds", 2)
        if bounds[0] > bounds[1]:
            raise ValueError(tr_("Invalid anomaly model scoring state"))
        payloads = {}
        for name in expected - {"metadata.json"}:
            if archive.getinfo(name).file_size > legacy.MAX_PAYLOAD_BYTES:
                raise ValueError(tr_("Anomaly model exceeds the supported size limit"))
            payloads[name] = archive.read(name)
            if hashlib.sha256(payloads[name]).hexdigest() != metadata["payloads"][name]:
                raise ValueError(tr_("Anomaly model payload checksum mismatch"))
        legacy._json_bytes(metadata)
        state = json.loads(payloads["state.json"])
        legacy._json_bytes(state)
        return metadata, payloads


def deserialize_bundle(metadata, payloads):
    """Called only after the public loader has checked explicit trust."""
    import joblib
    artifact = joblib.load(io.BytesIO(payloads["model.joblib"]))
    from .nodes.anomaly import AnomalyModelArtifact
    if not isinstance(artifact, AnomalyModelArtifact) or artifact.algorithm != metadata["algorithm"]:
        raise ValueError(tr_("Anomaly model metadata does not match scoring state"))
    if artifact.algorithm == "autoencoder":
        import tensorflow as tf
        with tempfile.TemporaryDirectory(prefix="anomaly-load-") as directory:
            network = Path(directory) / "network.keras"
            network.write_bytes(payloads["network.keras"])
            artifact.estimator = tf.keras.models.load_model(network, compile=False, safe_mode=True)
    legacy.validate_model(artifact)
    expected = legacy._model_metadata(artifact)
    expected["format_version"] = 2
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError(tr_("Anomaly model metadata does not match scoring state"))
    state = json.loads(payloads["state.json"])
    actual = {"preprocessing": artifact.preprocessing_state.to_dict() if artifact.preprocessing_state else None,
              "reference_parent_ids": artifact.reference_parent_ids,
              "reference_sample_ids": artifact.reference_sample_ids}
    if state != actual:
        raise ValueError(tr_("Anomaly preprocessing metadata does not match scoring state"))
    return artifact
