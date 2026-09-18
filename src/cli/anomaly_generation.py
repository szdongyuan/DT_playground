"""Native anomaly workflow templates shared by CLI and GUI execution."""

from pathlib import Path

from .generation import _node, _connection, _output_path, _validate_then_publish, _portable_reference
from .validation import validate_workflow_file
from .contracts import CLI_SCHEMA_VERSION


def create_anomaly_workflow(output, dataset, *, phase="train", algorithm="knn", model=None,
                            validation_dataset=None, aggregation="top_fraction_mean", k=5,
                            epochs=40, overwrite=False, trust_model=False, labels=None, protocol="generic"):
    output = _output_path(output, overwrite)
    dataset = Path(dataset).resolve()
    if not dataset.is_dir():
        raise ValueError("Audio dataset directory does not exist")
    if phase not in ("train", "score") or algorithm not in ("knn", "autoencoder"):
        raise ValueError("Unsupported anomaly template")
    if phase == "score" or algorithm == "autoencoder":
        if not model or not Path(model).is_file():
            raise ValueError("An existing model artifact is required")
    if phase == "score" and not trust_model:
        raise ValueError("Scoring requires --trust-model for an explicitly trusted anomaly bundle")
    if labels and (phase != "score" or not Path(labels).is_file()):
        raise ValueError("Evaluation labels require scoring mode and an existing mapping file")
    if phase == "train" and algorithm == "autoencoder":
        if not validation_dataset or not Path(validation_dataset).is_dir():
            raise ValueError("Autoencoder training requires a separate normal validation directory")
        val_directory = Path(validation_dataset).resolve()
        if val_directory == dataset or dataset in val_directory.parents or val_directory in dataset.parents:
            raise ValueError("Training and validation directories must be disjoint, including subdirectories")
    nodes, edges = [], []
    def node(name, kind, **params):
        nodes.append(_node(name, kind, len(nodes) * 180, 0, params))
    def link(source, port, target, input_port):
        edges.append(_connection(source, port, target, input_port))
    def features(prefix, folder, fit):
        node(prefix + "audio", "audio_folder", folder_path=_portable_reference(Path(folder).resolve(), output.parent),
             target_sr=16000, auto_label=False)
        node(prefix + "mel", "mel_spectrogram", n_mels=64, n_fft=1024, hop_length=512, power_to_db=True)
        node(prefix + "standard", "feature_standardization", mode="fit_transform" if fit else "transform")
        node(prefix + "windows", "feature_vectorizer", mode="sliding_window", aggregation="flatten",
             window_length=5, window_stride=5, tail_policy="include_last", short_policy="error")
        link(prefix + "audio", "audio", prefix + "mel", "audio")
        link(prefix + "mel", "feature", prefix + "standard", "features")
        link(prefix + "standard", "features", prefix + "windows", "features")
    features("", dataset, phase == "train")
    if phase == "score":
        node("model", "load_anomaly_model", model_path=_portable_reference(Path(model).resolve(), output.parent), trusted=True)
        link("model", "preprocessing_state", "standard", "state")
        source = "model"
    elif algorithm == "knn":
        node("trainer", "anomaly_detector_trainer", algorithm="knn", scaling="none", n_neighbors=k)
        link("windows", "feature_matrix", "trainer", "feature_matrix")
        link("standard", "state", "trainer", "preprocessing_state")
        node("save", "save_anomaly_model")
        link("trainer", "anomaly_model", "save", "anomaly_model")
        source = "save"
    else:
        features("val_", validation_dataset, False)
        link("standard", "state", "val_standard", "state")
        node("network", "load_model", model_path=_portable_reference(Path(model).resolve(), output.parent))
        node("trainer", "regression_trainer", epochs=epochs, batch_size=256, patience=5)
        link("network", "model", "trainer", "model")
        for port in ("x_train", "y_train"):
            link("windows", "feature_matrix", "trainer", port)
        for port in ("x_val", "y_val"):
            link("val_windows", "feature_matrix", "trainer", port)
        node("save", "save_anomaly_model", mode="autoencoder")
        link("trainer", "trained_model", "save", "model")
        link("windows", "feature_matrix", "save", "reference_features")
        link("standard", "state", "save", "preprocessing_state")
        source = "save"
    node("score", "anomaly_scorer", aggregation=aggregation)
    link(source, "anomaly_model", "score", "anomaly_model")
    link("windows", "feature_matrix", "score", "feature_matrix")
    node("export", "export_anomaly_results")
    link("score", "anomaly_scores", "export", "anomaly_scores")
    if labels:
        node("labels", "target_file", file_path=_portable_reference(Path(labels).resolve(), output.parent),
             target_kind="continuous", dtype="int64", output_format="map")
        node("evaluate", "anomaly_evaluation", protocol=protocol)
        link("labels", "target_map", "evaluate", "labels")
        link("score", "anomaly_scores", "evaluate", "anomaly_scores")
    definition = {"version": "1.0", "metadata": {"name": f"{algorithm}_{phase}",
                  "description": "Normal-only anomaly workflow. Labels are evaluation-only."},
                  "nodes": nodes, "connections": edges}
    report = _validate_then_publish(output, definition, lambda path: validate_workflow_file(path, headless=True))
    return {"schema_version": CLI_SCHEMA_VERSION, "summary": f"Anomaly workflow created: {output}",
            "output": str(output), "phase": phase, "algorithm": algorithm, "validation": report.to_dict(),
            "requires_model_trust": phase == "score"}
