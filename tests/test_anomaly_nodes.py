import numpy as np

from src.workflow.node_base import create_node
from src.workflow.nodes.anomaly import (
    AnomalyModelArtifact,
    AnomalyResultData,
    AnomalyScoresData,
    FeatureMatrixData,
)
from src.workflow.nodes.feature import FeatureData
from src.workflow.port import DataType


def _feature(data, source):
    return FeatureData(
        data=np.asarray(data, dtype=np.float32),
        feature_type="test",
        sample_rate=16000,
        hop_length=160,
        source_file=source,
    )


def test_anomaly_nodes_expose_typed_ports():
    vectorizer = create_node("feature_vectorizer")
    trainer = create_node("anomaly_detector_trainer")
    scorer = create_node("anomaly_scorer")
    decision = create_node("anomaly_decision")
    explorer = create_node("anomaly_explorer")

    assert vectorizer.outputs["feature_matrix"].data_type == DataType.FEATURE_MATRIX
    assert trainer.outputs["anomaly_model"].data_type == DataType.ANOMALY_MODEL
    assert scorer.outputs["anomaly_scores"].data_type == DataType.ANOMALY_SCORES
    assert decision.outputs["anomaly_result"].data_type == DataType.ANOMALY_RESULT
    assert explorer.inputs["anomaly_result"].data_type == DataType.ANOMALY_RESULT


def test_feature_vectorizer_preserves_alignment_and_reduces_last_axis():
    node = create_node("feature_vectorizer")
    node.inputs["features"].data = [
        _feature([[[1, 2, 3], [4, 5, 6]]], "a.wav"),
        _feature([[[2, 3, 4], [5, 6, 7]]], "b.wav"),
    ]

    assert node.execute(), node.error_message
    result = node.outputs["feature_matrix"].data

    assert isinstance(result, FeatureMatrixData)
    assert result.matrix.shape == (2, 4)
    assert result.sample_ids == ["a.wav", "b.wav"]
    assert result.source_items[0].source_file == "a.wav"
    np.testing.assert_allclose(result.matrix[0, :2], [2.0, 5.0])


def test_feature_vectorizer_rejects_inconsistent_flattened_sizes():
    node = create_node("feature_vectorizer")
    node.set_parameter("aggregation", "flatten")
    node.inputs["features"].data = [np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0])]

    assert node.execute() is False
    assert "inconsistent" in node.error_message.lower()


def test_isolation_forest_training_scoring_and_serialization(tmp_path):
    rng = np.random.default_rng(42)
    normal = rng.normal(0.0, 0.15, size=(30, 2))
    matrix = np.vstack([normal, np.array([[6.0, 6.0]])])
    features = FeatureMatrixData(
        matrix=matrix,
        sample_ids=[f"sample_{index}" for index in range(len(matrix))],
        source_items=list(range(len(matrix))),
        schema={"aggregation": "flatten", "feature_count": 2},
    )

    trainer = create_node("anomaly_detector_trainer")
    trainer.set_parameter("n_estimators", 50)
    trainer.set_parameter("contamination", 0.05)
    trainer.inputs["feature_matrix"].data = features

    assert trainer.execute(), trainer.error_message
    artifact = trainer.outputs["anomaly_model"].data
    reference_scores = trainer.outputs["reference_scores"].data

    assert isinstance(artifact, AnomalyModelArtifact)
    assert isinstance(reference_scores, AnomalyScoresData)
    assert reference_scores.raw_scores[-1] > np.median(reference_scores.raw_scores[:-1])
    assert trainer.outputs["training_summary"].data["sample_count"] == 31

    path = tmp_path / "detector.anomaly.zip"
    artifact.save(str(path))
    loaded = AnomalyModelArtifact.load(str(path), trusted=True)
    np.testing.assert_allclose(loaded.score(matrix), artifact.score(matrix))

    scorer = create_node("anomaly_scorer")
    scorer.inputs["anomaly_model"].data = loaded
    scorer.inputs["feature_matrix"].data = features
    assert scorer.execute(), scorer.error_message
    assert len(scorer.outputs["anomaly_scores"].data.normalized_scores) == 31


def test_anomaly_scorer_rejects_feature_count_mismatch():
    training_features = FeatureMatrixData(
        np.array([[0.0, 0.0], [0.1, 0.1], [4.0, 4.0]]),
        ["a", "b", "c"],
    )
    trainer = create_node("anomaly_detector_trainer")
    trainer.set_parameter("n_estimators", 10)
    trainer.inputs["feature_matrix"].data = training_features
    assert trainer.execute(), trainer.error_message

    scorer = create_node("anomaly_scorer")
    scorer.inputs["anomaly_model"].data = trainer.outputs["anomaly_model"].data
    scorer.inputs["feature_matrix"].data = FeatureMatrixData(
        np.array([[0.0, 0.0, 0.0]]),
        ["mismatch"],
    )

    assert scorer.execute() is False
    assert "Feature count mismatch" in scorer.error_message


def test_manual_anomaly_decision_has_normal_attention_and_anomaly_boundaries():
    scores = AnomalyScoresData(
        raw_scores=np.array([0.1, 0.7, 0.9]),
        normalized_scores=np.array([10.0, 70.0, 90.0]),
        sample_ids=["normal", "attention", "anomaly"],
        reference_normalized_scores=np.array([5.0, 10.0, 20.0]),
    )
    node = create_node("anomaly_decision")
    node.set_parameter("strategy", "manual")
    node.set_parameter("manual_threshold", 80.0)
    node.set_parameter("attention_ratio", 0.75)
    node.inputs["anomaly_scores"].data = scores

    assert node.execute(), node.error_message
    result = node.outputs["anomaly_result"].data

    assert isinstance(result, AnomalyResultData)
    assert result.attention_threshold == 60.0
    assert result.severities == ["normal", "attention", "anomaly"]
    assert result.is_anomaly.tolist() == [False, False, True]


def test_reference_quantile_handles_identical_reference_scores_without_flagging_zero():
    scores = AnomalyScoresData(
        raw_scores=np.array([0.0]),
        normalized_scores=np.array([0.0]),
        sample_ids=["normal"],
        reference_normalized_scores=np.array([0.0, 0.0, 0.0]),
    )
    node = create_node("anomaly_decision")
    node.inputs["anomaly_scores"].data = scores

    assert node.execute(), node.error_message
    result = node.outputs["anomaly_result"].data
    assert result.threshold > 0.0
    assert result.is_anomaly.tolist() == [False]


def test_anomaly_explorer_passes_result_to_preview_output():
    scores = AnomalyScoresData(
        raw_scores=np.array([0.1]),
        normalized_scores=np.array([10.0]),
        sample_ids=["sample"],
    )
    result = AnomalyResultData(
        scores=scores,
        threshold=80.0,
        attention_threshold=60.0,
        severities=["normal"],
        is_anomaly=np.array([False]),
        strategy="manual",
    )
    node = create_node("anomaly_explorer")
    node.inputs["anomaly_result"].data = result

    assert node.execute(), node.error_message
    assert node.outputs["anomaly_result"].data is result
