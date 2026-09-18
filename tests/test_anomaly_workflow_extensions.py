"""Numerical and contract regressions for native anomaly workflows."""

import json
import zipfile

import numpy as np
import pytest

from src.workflow.node_base import create_node
from src.workflow.nodes.anomaly import FeatureMatrixData, AnomalyScoresData
from src.workflow.nodes.feature import FeatureData
from src.workflow.feature_contract import standardize, aggregate_scores
from src.workflow.anomaly_backends import KNNReference
from src.workflow.anomaly_io import save_model, load_model, inspect_model
from src.workflow.nodes.anomaly_evaluation import evaluate_anomalies


def feature(data, name="a.wav"):
    return FeatureData(np.asarray(data, dtype=float), "mel", 16000, 512, name)


def execute(kind, inputs, **parameters):
    node = create_node(kind)
    for key, value in parameters.items():
        assert node.set_parameter(key, value)[0]
    for key, value in inputs.items():
        node.inputs[key].data = value
    assert node.execute(), node.error_message
    return {key: port.data for key, port in node.outputs.items()}


def test_temporal_standardization_fits_frames_once_and_never_refits():
    data = [feature([[1, 3, 5], [2, 2, 2]]), feature([[7], [2]], "b.wav")]
    normalized, state = standardize(data)
    np.testing.assert_allclose(state.mean, [4, 2])
    np.testing.assert_allclose(state.scale, [np.sqrt(5), 1e-6])
    before = state.fingerprint
    validation, reused = standardize([feature([[100], [3]])], state)
    assert state.fingerprint == before == reused.fingerprint
    np.testing.assert_allclose(validation[0].data[:, 0], [(100 - 4) / np.sqrt(5), 1e6])
    assert normalized[0].metadata["preprocessing"] == before
    with pytest.raises(ValueError, match="layout"):
        standardize([FeatureData(np.ones((2, 2)), "mfcc", 16000, 512)], state)


def test_window_tail_padding_and_distinct_parent_identity():
    data = [feature(np.arange(14).reshape(2, 7), "one/a.wav"), feature(np.arange(14).reshape(2, 7), "two/a.wav")]
    result = execute("feature_vectorizer", {"features": data}, mode="sliding_window",
                     aggregation="flatten", window_length=3, window_stride=3)["feature_matrix"]
    assert result.matrix.shape == (6, 6)
    assert [row["start_frame"] for row in result.provenance[:3]] == [0, 3, 4]
    assert len(set(result.sample_ids)) == 6
    assert len(set(result.parent_ids)) == 2
    np.testing.assert_array_equal(result.matrix[2], [4, 5, 6, 11, 12, 13])
    padded = execute("feature_vectorizer", {"features": [feature([[1, 2]])]},
                     mode="sliding_window", aggregation="flatten", window_length=3, short_policy="pad_edge")["feature_matrix"]
    np.testing.assert_array_equal(padded.matrix, [[1, 2, 2]])
    assert padded.provenance[0]["padded_frames"] == 1
    assert padded.provenance[0]["end_frame"] == 2


def test_knn_excludes_identity_not_equal_values_and_same_file_windows():
    bank = KNNReference(np.array([[0.], [0.], [3.], [10.]]), ["a", "b", "c", "d"], ["f", "f", "g", "h"], k=1)
    np.testing.assert_allclose(bank.score(bank.matrix, bank.sample_ids, bank.parent_ids), [3, 3, 3, 7])
    bank.exclusion = "sample"
    np.testing.assert_allclose(bank.score(bank.matrix, bank.sample_ids, bank.parent_ids), [0, 0, 3, 7])
    bank.k = 4
    with pytest.raises(ValueError, match="Insufficient"):
        bank.score(bank.matrix, bank.sample_ids, bank.parent_ids)
    bank.metric = "cosine"
    with pytest.raises(ValueError, match="zero"):
        bank.validate()


def test_aggregation_uses_raw_tail_ceil_and_recalibrates_reference(tmp_path):
    matrix = FeatureMatrixData(np.array([[0.], [1.], [4.], [6.]]), list("abcd"),
                              schema={"feature_count": 1}, provenance=[{"parent_sample_id": p} for p in ["f", "f", "g", "g"]])
    model = execute("anomaly_detector_trainer", {"feature_matrix": matrix},
                    algorithm="knn", n_neighbors=1, scaling="none")["anomaly_model"]
    scored = execute("anomaly_scorer", {"feature_matrix": matrix, "anomaly_model": model}, aggregation="mean")
    expected, _, _ = aggregate_scores(model.reference_scores, matrix.parent_ids, "mean")
    np.testing.assert_allclose(scored["anomaly_scores"].raw_scores, expected)
    assert scored["anomaly_scores"].sample_ids == ["f", "g"]
    assert len(scored["detailed_scores"].sample_ids) == 4
    assert len(scored["anomaly_scores"].reference_normalized_scores) == 2
    assert aggregate_scores([1, 2, 100], ["f"] * 3, "top_fraction_mean", .34)[0] == 51
    path = tmp_path / "knn.anomaly.zip"
    save_model(model, path)
    assert inspect_model(path)["format_version"] == 2
    loaded = load_model(path, trusted=True)
    np.testing.assert_array_equal(loaded.score(matrix.matrix, matrix.sample_ids, matrix.parent_ids), model.reference_scores)
    with pytest.raises(PermissionError):
        load_model(path)


def test_strict_schema_double_scaling_and_calibration_overlap():
    matrix, state = standardize(FeatureMatrixData(np.array([[1.], [2.], [3.]]), list("abc")))
    node = create_node("anomaly_detector_trainer")
    node.inputs["feature_matrix"].data = matrix
    assert not node.execute()
    assert "scaling=none" in node.error_message
    node.set_parameter("scaling", "none")
    node.inputs["calibration_features"].data = matrix
    assert not node.execute()
    assert "overlap" in node.error_message
    model = execute("anomaly_detector_trainer", {"feature_matrix": matrix}, scaling="none")["anomaly_model"]
    matrix.schema = {"different": True}
    scorer = create_node("anomaly_scorer")
    scorer.inputs["feature_matrix"].data = matrix
    scorer.inputs["anomaly_model"].data = model
    assert not scorer.execute()
    assert "schema" in scorer.error_message


def test_evaluation_uses_raw_scores_and_dcase_all_anomalies():
    from sklearn.metrics import roc_auc_score
    ids = [f"section_00_{domain}_test_{index}.wav" for index, domain in enumerate(["source", "source", "target", "target"])]
    raw = np.array([1., 3., 2., 4.])
    scores = AnomalyScoresData(raw, np.ones(4) * 100, ids)
    labels = dict(zip(ids, [0, 1, 0, 1]))
    result = evaluate_anomalies(scores, labels, "dcase")
    assert result["roc_auc"] == 1
    assert result["sections"]["00"]["auc_source"] == roc_auc_score([0, 1, 1], [1, 3, 4])
    with pytest.raises(ValueError, match="identities"):
        evaluate_anomalies(scores, {**labels, "extra": 1})
    scores.metadata["granularity"] = "window"
    with pytest.raises(ValueError, match="Aggregate"):
        evaluate_anomalies(scores, labels)


def test_ae_packaging_reload_preprocessing_and_batch_parity(tmp_path):
    tf = pytest.importorskip("tensorflow")
    tf.keras.utils.set_random_seed(42)
    model = tf.keras.Sequential([tf.keras.Input((2,)), tf.keras.layers.Dense(2)])
    matrix, state = standardize(FeatureMatrixData(np.array([[1., 2.], [3., 5.], [4., 9.]]), list("abc")))
    outputs = execute("save_anomaly_model", {"model": model, "reference_features": matrix,
                      "preprocessing_state": state}, mode="autoencoder", output_folder=str(tmp_path))
    artifact = outputs["anomaly_model"]
    loaded = load_model(outputs["model_path"], trusted=True)
    np.testing.assert_allclose(artifact.score(matrix.matrix, batch_size=1), loaded.score(matrix.matrix, batch_size=2), rtol=1e-6)
    assert loaded.preprocessing_state.fingerprint == state.fingerprint
    with zipfile.ZipFile(outputs["model_path"]) as archive:
        assert "network.keras" in archive.namelist()
    from src.workflow.nodes.training import RegressionTrainerNode, RegressionEvaluatorNode, PredictNode, _target_validation_reason
    assert _target_validation_reason("regression", matrix) is None
    for node in (RegressionTrainerNode(), RegressionEvaluatorNode(), PredictNode()):
        np.testing.assert_array_equal(node._convert_to_array(matrix, model), matrix.matrix)


def test_best_epoch_restores_matching_optimizer(tmp_path):
    tf = pytest.importorskip("tensorflow")
    from src.training.consistent_early_stopping import ConsistentEarlyStopping
    model = tf.keras.Sequential([tf.keras.Input((1,)), tf.keras.layers.Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    callback = ConsistentEarlyStopping(monitor="loss", patience=1, restore_best_weights=True)
    callback.set_model(model)
    callback.on_train_begin()
    model.train_on_batch(np.array([[1.]]), np.array([[0.]]))
    weights = model.get_weights()
    optimizer = [x.numpy().copy() for x in model.optimizer.variables]
    callback.on_epoch_end(0, {"loss": 1.})
    model.train_on_batch(np.array([[1.]]), np.array([[10.]]))
    callback.on_epoch_end(1, {"loss": 2.})
    callback.on_train_end()
    for expected, actual in zip(weights, model.get_weights()):
        np.testing.assert_array_equal(expected, actual)
    for expected, actual in zip(optimizer, model.optimizer.variables):
        np.testing.assert_array_equal(expected, actual)
    path = tmp_path / "best.keras"
    model.save(path)
    resumed = tf.keras.models.load_model(path)
    assert int(resumed.optimizer.iterations.numpy()) == 1
    resumed.train_on_batch(np.array([[1.]]), np.array([[0.]]))
    assert int(resumed.optimizer.iterations.numpy()) == 2


@pytest.mark.parametrize("algorithm", ["knn", "autoencoder"])
def test_native_cli_train_reload_and_gui_parity(tmp_path, qapp, algorithm):
    import soundfile as sf

    from src.cli.main import main
    from src.workflow import Workflow
    from src.workflow.engine import WorkflowEngine
    folder, validation = tmp_path / "train_audio", tmp_path / "val_audio"
    folder.mkdir()
    validation.mkdir()
    rng = np.random.default_rng(42)
    for root, count in ((folder, 4), (validation, 2)):
        for index in range(count):
            sf.write(root / f"{index}.wav", rng.normal(0, .1, 4000), 16000)
    source = tmp_path / "train.workflow.json"
    arguments = ["create-anomaly-workflow", str(source), "--dataset", str(folder), "--algorithm", algorithm, "--k", "1"]
    if algorithm == "autoencoder":
        tf = pytest.importorskip("tensorflow")
        model = tf.keras.Sequential([tf.keras.Input((320,)), tf.keras.layers.Dense(4), tf.keras.layers.Dense(320)])
        model.compile(optimizer="adam", loss="mse")
        initial = tmp_path / "initial.keras"
        model.save(initial)
        arguments.extend(["--model", str(initial), "--validation-dataset", str(validation), "--epochs", "1"])
    assert main(arguments) == 0
    assert main(["run", str(source), "--run-dir", str(tmp_path / "train")]) == 0
    bundle = tmp_path / "train/anomaly_model.anomaly.zip"
    scoring = tmp_path / "score.workflow.json"
    assert main(["create-anomaly-workflow", str(scoring), "--dataset", str(folder), "--phase", "score",
                 "--model", str(bundle), "--trust-model"]) == 0
    assert main(["validate", str(scoring), "--check-data", "--json"]) == 0
    assert main(["run", str(scoring), "--run-dir", str(tmp_path / "score")]) == 0
    assert (tmp_path / "train/anomaly_results/results.csv").read_bytes() == (tmp_path / "score/anomaly_results/results.csv").read_bytes()
    graph = Workflow.load(scoring)
    graph.nodes["audio"].set_parameter("folder_path", str(folder))
    graph.nodes["model"].set_parameter("model_path", str(bundle))
    graph.nodes["export"].set_parameter("output_folder", str(tmp_path / "gui"))
    engine = WorkflowEngine()
    engine.set_workflow(graph)
    result = engine.execute_sync()
    assert result.success, result.message
    assert (tmp_path / "gui/anomaly_results/results.csv").read_bytes() == (tmp_path / "score/anomaly_results/results.csv").read_bytes()


def test_legacy_if_schema_remains_usable():
    matrix = execute("feature_vectorizer", {"features": [feature([[1, 2]], "a"), feature([[3, 5]], "b")]})["feature_matrix"]
    model = execute("anomaly_detector_trainer", {"feature_matrix": matrix})["anomaly_model"]
    model.feature_schema = {"aggregation": "mean_std", "feature_count": 2}
    result = execute("anomaly_scorer", {"feature_matrix": matrix, "anomaly_model": model})
    assert len(result["anomaly_scores"].raw_scores) == 2


def test_typed_training_inputs_reject_identity_schema_and_split_leakage():
    from dataclasses import replace

    from src.workflow.nodes.training import _validate_feature_matrix_inputs
    matrix = FeatureMatrixData(np.ones((2, 2)), ["a", "b"], schema={"feature_count": 2})
    with pytest.raises(ValueError, match="identities"):
        _validate_feature_matrix_inputs(matrix, replace(matrix, sample_ids=["b", "a"]))
    with pytest.raises(ValueError, match="schemas"):
        _validate_feature_matrix_inputs(matrix, matrix, replace(matrix, sample_ids=["c", "d"], schema={}))
    with pytest.raises(ValueError, match="parent"):
        _validate_feature_matrix_inputs(matrix, matrix, matrix, matrix)
    normalized, _ = standardize(matrix)
    with pytest.raises(ValueError, match="already standardized"):
        standardize(normalized)


def test_template_rejects_nested_validation_directory(tmp_path):
    from src.cli.anomaly_generation import create_anomaly_workflow
    child = tmp_path / "validation"
    child.mkdir()
    model = tmp_path / "model.keras"
    model.touch()
    with pytest.raises(ValueError, match="disjoint"):
        create_anomaly_workflow(tmp_path / "flow.json", tmp_path, algorithm="autoencoder", model=model,
                                validation_dataset=child)


@pytest.mark.parametrize("file_format,content", [("json", '{"a": 0, "a": 1}'), ("csv", 'filename,target\na,0\na,1\n')])
def test_duplicate_label_file_identity_is_rejected(tmp_path, file_format, content):
    path = tmp_path / f"labels.{file_format}"
    path.write_text(content)
    node = create_node("target_file")
    node.set_parameter("file_path", str(path))
    assert not node.execute()
    assert "Duplicate target identity" in node.error_message


@pytest.mark.parametrize("fault", ["hash", "version", "path", "duplicate", "size"])
def test_v2_archive_rejects_corruption_before_deserialization(tmp_path, monkeypatch, fault):
    import src.workflow.anomaly_io as persistence
    matrix = FeatureMatrixData(np.array([[0.], [1.], [2.]]), list("abc"))
    model = execute("anomaly_detector_trainer", {"feature_matrix": matrix},
                    algorithm="knn", n_neighbors=1, scaling="none")["anomaly_model"]
    path = tmp_path / "good.zip"
    save_model(model, path)
    with zipfile.ZipFile(path) as archive:
        payloads = {name: archive.read(name) for name in archive.namelist()}
    metadata = json.loads(payloads["metadata.json"])
    if fault == "hash":
        payloads["model.joblib"] += b"corrupt"
    elif fault == "version":
        metadata["format_version"] = 99
    elif fault == "path":
        payloads["../extra"] = b"unsafe"
    elif fault == "size":
        monkeypatch.setattr(persistence, "MAX_PAYLOAD_BYTES", 1)
    payloads["metadata.json"] = json.dumps(metadata).encode()
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as archive:
        for name, data in payloads.items():
            archive.writestr(name, data)
        if fault == "duplicate":
            archive.writestr("state.json", payloads["state.json"])
    import joblib
    monkeypatch.setattr(joblib, "load", lambda *a, **k: pytest.fail("Must reject archive before deserialization"))
    with pytest.raises(ValueError):
        load_model(bad, trusted=True)
