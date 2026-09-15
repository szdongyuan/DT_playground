import numpy as np
import pytest

from src.workflow.node_base import create_node
from src.workflow.port import DataType


class _ClassificationModel:
    def __init__(self, probabilities, metrics=None):
        self.probabilities = np.asarray(probabilities, dtype=np.float32)
        self.metrics = metrics or {"loss": 0.25}

    def evaluate(self, _data, _targets, batch_size, verbose, return_dict):
        assert batch_size == 32
        assert verbose == 0
        assert return_dict is True
        return self.metrics

    def predict(self, _data, batch_size, verbose):
        assert batch_size == 32
        assert verbose == 0
        return self.probabilities


def _node(probabilities, targets, metadata=None, file_paths=None):
    node = create_node("classification_evaluator")
    node.inputs["model"].data = _ClassificationModel(probabilities)
    node.inputs["x_test"].data = np.arange(len(targets), dtype=np.float32)[:, None]
    node.inputs["y_test"].data = np.asarray(targets)
    node.inputs["target_metadata"].data = metadata
    if file_paths is not None:
        node.inputs["file_paths"].data = file_paths
    return node


def test_classification_evaluator_replaces_metrics_with_structured_result():
    node = _node(
        probabilities=[
            [0.8, 0.1, 0.1],
            [0.1, 0.7, 0.2],
            [0.1, 0.6, 0.3],
            [0.1, 0.2, 0.7],
        ],
        targets=[0, 1, 2, 2],
        metadata={
            "kind": "categorical",
            "category_mapping": {"neutral": 0, "happy": 1, "sad": 2},
        },
        file_paths=["a.wav", "b.wav", "c.wav", "d.wav"],
    )

    assert list(node.outputs) == ["classification_result"]
    assert node.outputs["classification_result"].data_type == DataType.CLASSIFICATION_RESULT
    statuses = []
    node.status_callback = statuses.append
    assert node.execute(), node.error_message

    result = node.outputs["classification_result"].data
    assert result["schema_version"] == "1.0"
    assert result["task"] == "classification"
    assert result["sample_count"] == 4
    assert result["classes"] == [
        {"id": 0, "name": "neutral"},
        {"id": 1, "name": "happy"},
        {"id": 2, "name": "sad"},
    ]
    assert result["metrics"]["loss"] == pytest.approx(0.25)
    assert result["metrics"]["accuracy"] == pytest.approx(0.75)
    assert result["metrics"]["balanced_accuracy"] == pytest.approx(5 / 6)
    assert result["metrics"]["macro_f1"] == pytest.approx(7 / 9)
    assert result["metrics"]["weighted_f1"] == pytest.approx(0.75)
    assert result["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 1, 1]]
    assert result["per_class"][2] == {
        "class_id": 2,
        "class_name": "sad",
        "precision": pytest.approx(1.0),
        "recall": pytest.approx(0.5),
        "f1": pytest.approx(2 / 3),
        "support": 2,
    }
    assert result["predictions"][2] == {
        "index": 2,
        "file_path": "c.wav",
        "true_class_id": 2,
        "true_class_name": "sad",
        "predicted_class_id": 1,
        "predicted_class_name": "happy",
        "confidence": pytest.approx(0.6),
    }
    assert result["warnings"] == []
    assert "balanced_accuracy=0.8333" in statuses[-1]
    assert "confusion_matrix=[[1, 0, 0], [0, 1, 0], [0, 1, 1]]" in statuses[-1]


def test_classification_evaluator_reports_classes_absent_from_test_and_predictions():
    node = _node(
        probabilities=[[0.8, 0.2, 0.0], [0.7, 0.3, 0.0]],
        targets=[0, 0],
        metadata={
            "kind": "categorical",
            "category_mapping": {"zero": 0, "one": 1, "two": 2},
        },
    )

    assert node.execute(), node.error_message
    result = node.outputs["classification_result"].data
    assert result["per_class"][1]["support"] == 0
    assert result["per_class"][2]["support"] == 0
    assert result["warnings"] == [
        {"code": "missing_true_classes", "class_ids": [1, 2]},
        {"code": "unpredicted_classes", "class_ids": [1, 2]},
    ]


def test_classification_evaluator_supports_binary_sigmoid_predictions():
    node = _node(
        probabilities=[[0.2], [0.8], [0.49], [0.51]],
        targets=[0, 1, 0, 1],
        metadata={"kind": "categorical"},
    )

    assert node.execute(), node.error_message
    result = node.outputs["classification_result"].data
    assert result["confusion_matrix"] == [[2, 0], [0, 2]]
    assert [row["confidence"] for row in result["predictions"]] == pytest.approx(
        [0.8, 0.8, 0.51, 0.51]
    )


def test_classification_evaluator_rejects_mismatched_file_path_count():
    node = _node(
        probabilities=[[0.8, 0.2], [0.2, 0.8]],
        targets=[0, 1],
        metadata={"kind": "categorical"},
        file_paths=["only-one.wav"],
    )

    assert node.execute() is False
    assert "File path count" in node.error_message


def test_classification_evaluator_bounds_large_confusion_matrix_in_status():
    class_count = 11
    node = _node(
        probabilities=np.eye(class_count, dtype=np.float32),
        targets=list(range(class_count)),
        metadata={"kind": "categorical"},
    )
    statuses = []
    node.status_callback = statuses.append

    assert node.execute(), node.error_message
    assert "confusion_matrix=shape=11x11" in statuses[-1]
    assert len(node.outputs["classification_result"].data["confusion_matrix"]) == 11
