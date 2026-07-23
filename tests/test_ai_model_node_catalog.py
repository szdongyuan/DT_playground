import numpy as np
from PySide6.QtCore import Qt

from src.ui.i18n import tr_
from src.ui.node_editor.node_palette import NodePalette
from src.workflow.node_base import NodeCategory, create_node, get_nodes_by_category
from src.workflow.port import DataType


class _PredictModel:
    def __init__(self, predictions):
        self._predictions = predictions

    def predict(self, _data, batch_size, verbose):
        assert batch_size == 32
        assert verbose == 0
        return self._predictions


class _EvaluateModel:
    metrics_names = ["loss", "metric"]

    def evaluate(self, _data, _targets, batch_size, verbose):
        assert batch_size == 32
        assert verbose == 0
        return [0.25, 0.75]


def _set_input(node, name, value):
    node.inputs[name].data = value


def _visible_types(category):
    return [node_class.node_type for node_class in get_nodes_by_category(category)]


def test_legacy_task_nodes_remain_loadable_but_hidden_from_palette():
    for node_type in ("trainer", "evaluator", "predict"):
        assert create_node(node_type) is not None
        assert node_type not in _visible_types(NodeCategory.TRAINING)


def test_ai_model_nodes_use_stage_based_subcategories_and_order():
    expected = {
        "classification_trainer": ("Model training", 10, 10),
        "regression_trainer": ("Model training", 10, 20),
        "anomaly_detector_trainer": ("Model training", 10, 30),
        "classification_predict": ("Inference and decision", 20, 10),
        "regression_predict": ("Inference and decision", 20, 20),
        "anomaly_scorer": ("Inference and decision", 20, 30),
        "anomaly_decision": ("Inference and decision", 20, 40),
        "classification_evaluator": ("Model evaluation", 30, 10),
        "regression_evaluator": ("Model evaluation", 30, 20),
        "load_model": ("Model management", 40, 10),
        "save_model": ("Model management", 40, 20),
    }

    assert NodeCategory.TRAINING.display_name == tr_("AI / Model")
    for node_type, (subcategory, subcategory_order, palette_order) in expected.items():
        node = create_node(node_type)
        assert node.subcategory == tr_(subcategory)
        assert node.subcategory_order == subcategory_order
        assert node.palette_order == palette_order


def test_palette_renders_stage_groups_and_nodes_in_workflow_order(qapp):
    palette = NodePalette()
    tree = palette._tree
    category_item = next(
        tree.topLevelItem(index)
        for index in range(tree.topLevelItemCount())
        if tree.topLevelItem(index).text(0) == NodeCategory.TRAINING.display_name
    )

    assert [category_item.child(index).text(0) for index in range(category_item.childCount())] == [
        f"📁 {tr_('Model training')}",
        f"📁 {tr_('Inference and decision')}",
        f"📁 {tr_('Model evaluation')}",
        f"📁 {tr_('Model management')}",
    ]

    expected_types = [
        ["classification_trainer", "regression_trainer", "anomaly_detector_trainer"],
        ["classification_predict", "regression_predict", "anomaly_scorer", "anomaly_decision"],
        ["classification_evaluator", "regression_evaluator"],
        ["load_model", "save_model"],
    ]
    for group_index, node_types in enumerate(expected_types):
        group = category_item.child(group_index)
        assert [
            group.child(index).data(0, Qt.ItemDataRole.UserRole)
            for index in range(group.childCount())
        ] == node_types


def test_visualization_nodes_are_grouped_under_output_category(qapp):
    assert NodeCategory.OUTPUT.display_name == tr_("Output / Visualization")
    palette = NodePalette()
    tree = palette._tree
    output_item = next(
        tree.topLevelItem(index)
        for index in range(tree.topLevelItemCount())
        if tree.topLevelItem(index).text(0) == NodeCategory.OUTPUT.display_name
    )
    output_types = [
        output_item.child(index).data(0, Qt.ItemDataRole.UserRole)
        for index in range(output_item.childCount())
    ]

    assert output_types == ["show_history", "show_metrics", "anomaly_explorer"]


def test_task_specific_evaluators_validate_target_semantics():
    classification = create_node("classification_evaluator")
    regression = create_node("regression_evaluator")

    assert classification.inputs["target_metadata"].required is False
    assert regression.inputs["target_metadata"].required is False

    for node in (classification, regression):
        _set_input(node, "model", _EvaluateModel())
        _set_input(node, "x_test", np.array([[1.0], [2.0]]))

    _set_input(classification, "y_test", np.array([0.1, 0.2]))
    _set_input(classification, "target_metadata", {"kind": "continuous"})
    assert classification.execute() is False
    assert "categorical" in classification.error_message

    _set_input(regression, "y_test", np.array([0.1, 0.2]))
    _set_input(regression, "target_metadata", {"kind": "continuous"})
    assert regression.execute() is True
    assert regression.outputs["metrics"].data == {"loss": 0.25, "metric": 0.75}


def test_classification_prediction_outputs_labels_and_unmodified_raw_values():
    raw = np.array([[0.1, 0.9], [0.8, 0.2]], dtype=np.float32)
    node = create_node("classification_predict")
    _set_input(node, "model", _PredictModel(raw))
    _set_input(node, "input_data", np.array([[1.0], [2.0]]))

    assert node.outputs["predicted_labels"].data_type == DataType.LABEL
    assert node.execute() is True
    assert node.outputs["predicted_labels"].data == [1, 0]
    assert node.outputs["raw_output"].data is raw


def test_regression_prediction_outputs_continuous_values():
    node = create_node("regression_predict")
    _set_input(node, "model", _PredictModel(np.array([[1.25], [2.5]])))
    _set_input(node, "input_data", np.array([[1.0], [2.0]]))

    assert node.execute() is True
    assert node.outputs["predicted_values"].data == [1.25, 2.5]
