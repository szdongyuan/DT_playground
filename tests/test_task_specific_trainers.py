import numpy as np

from src.workflow.node_base import NodeCategory, create_node, get_nodes_by_category


class _CompiledModel:
    def __init__(self, loss):
        self.loss = loss


def test_legacy_trainer_remains_loadable_but_is_hidden_from_palette():
    assert create_node("trainer") is not None

    visible_types = [
        node_class.node_type
        for node_class in get_nodes_by_category(NodeCategory.TRAINING)
    ]
    assert "trainer" not in visible_types
    assert "classification_trainer" in visible_types
    assert "regression_trainer" in visible_types


def test_task_specific_trainers_expose_target_metadata():
    classification = create_node("classification_trainer")
    regression = create_node("regression_trainer")

    assert classification.inputs["target_metadata"].required is False
    assert regression.inputs["target_metadata"].required is False
    assert classification.task_kind == "classification"
    assert regression.task_kind == "regression"


def test_classification_trainer_validates_categorical_targets():
    node = create_node("classification_trainer")

    assert node._validate_targets(np.array([0, 1, 2]), {})[0] is True
    assert node._validate_targets(np.array([0.0, 1.0, 2.0]), {})[0] is True
    assert node._validate_targets(np.eye(3), {})[0] is True
    assert node._validate_targets(
        np.array([0.1, 0.2, 0.3]),
        {"kind": "continuous"},
    )[0] is False


def test_regression_trainer_validates_continuous_targets():
    node = create_node("regression_trainer")

    assert node._validate_targets(
        np.array([0.1, 0.2, 0.3]),
        {"kind": "continuous"},
    )[0] is True
    assert node._validate_targets(
        np.array([0, 1, 2]),
        {"kind": "categorical"},
    )[0] is False
    assert node._validate_targets(["low", "high"], {})[0] is False


def test_task_specific_auto_loss_uses_task_semantics():
    classification = create_node("classification_trainer")
    regression = create_node("regression_trainer")

    assert classification._resolve_auto_loss(np.array([0, 1, 2])) == "sparse_categorical_crossentropy"
    assert regression._resolve_auto_loss(np.array([0.0, 15.5, 359.0])) == "mse"


def test_task_specific_trainers_reject_incompatible_model_losses():
    classification = create_node("classification_trainer")
    regression = create_node("regression_trainer")

    assert classification._validate_model_loss(_CompiledModel("mse"))[0] is False
    assert classification._validate_model_loss(
        _CompiledModel("sparse_categorical_crossentropy")
    )[0] is True
    assert regression._validate_model_loss(_CompiledModel("mse"))[0] is True
    assert regression._validate_model_loss(
        _CompiledModel("binary_crossentropy")
    )[0] is False
