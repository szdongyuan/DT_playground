import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from src.ui.views.model_builder_view import ModelBuilderView
from src.ui.views.workflow_editor_widget import WorkflowEditorWidget
from src.workflow import Workflow


def _choose_distinct_parameter_value(owner):
    for name, param in owner.parameters.items():
        default = owner.get_parameter(name)
        param_type = str(param.param_type.value if hasattr(param.param_type, "value") else param.param_type)

        if param_type == "choice" and param.choices:
            for choice in param.choices:
                if choice != default:
                    return name, choice
        elif param_type == "bool":
            return name, (not default)
        elif param_type == "int":
            candidate = default + 1 if default is not None else 1
            if param.max_value is not None and candidate > param.max_value:
                candidate = default - 1
            if param.min_value is None or candidate >= param.min_value:
                return name, candidate
        elif param_type == "float":
            candidate = float(default) + 0.5 if default is not None else 0.5
            if param.max_value is not None and candidate > param.max_value:
                candidate = float(default) - 0.5
            if param.min_value is None or candidate >= param.min_value:
                return name, candidate
        elif param_type == "str":
            return name, f"{default}_copy_test"

    raise AssertionError("No adjustable parameter found for test fixture")


class GraphCopyPasteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_workflow_copy_paste_preserves_parameters_and_internal_connections_only(self):
        widget = WorkflowEditorWidget()
        self.addCleanup(widget.close)
        widget.resize(900, 600)
        widget.show()

        workflow = Workflow("copy_paste_workflow")
        widget.set_workflow(workflow)

        node_graph = widget._node_graph
        source_id = node_graph.add_node("audio_file", (10.0, 20.0))
        middle_id = node_graph.add_node("resample", (160.0, 20.0))
        sink_id = node_graph.add_node("fft", (320.0, 20.0))

        self.assertIsNotNone(source_id)
        self.assertIsNotNone(middle_id)
        self.assertIsNotNone(sink_id)

        success, _ = workflow.connect(source_id, "audio", middle_id, "audio")
        self.assertTrue(success)
        node_graph._scene.add_connection_item(source_id, "audio", middle_id, "audio")

        success, _ = workflow.connect(middle_id, "audio", sink_id, "audio")
        self.assertTrue(success)
        node_graph._scene.add_connection_item(middle_id, "audio", sink_id, "audio")

        middle_node = workflow.get_node(middle_id)
        param_name, param_value = _choose_distinct_parameter_value(middle_node)
        success, message = middle_node.set_parameter(param_name, param_value)
        self.assertTrue(success, message)

        node_graph._scene.clearSelection()
        node_graph._scene.node_items[source_id].setSelected(True)
        node_graph._scene.node_items[middle_id].setSelected(True)
        self.app.processEvents()

        self.assertTrue(node_graph._copy_selected())
        self.assertTrue(node_graph._paste_clipboard())

        self.assertEqual(len(workflow.nodes), 5)
        self.assertEqual(len(workflow.connections), 3)

        original_ids = {source_id, middle_id, sink_id}
        new_nodes = {
            node_id: node
            for node_id, node in workflow.nodes.items()
            if node_id not in original_ids
        }
        self.assertEqual({node.node_type for node in new_nodes.values()}, {"audio_file", "resample"})

        copied_middle = next(node for node in new_nodes.values() if node.node_type == "resample")
        copied_source = next(node for node in new_nodes.values() if node.node_type == "audio_file")

        self.assertEqual(copied_middle.get_parameter(param_name), param_value)
        self.assertNotEqual(copied_middle.node_id, middle_id)
        self.assertNotEqual(copied_source.node_id, source_id)
        self.assertNotEqual(copied_middle.position, middle_node.position)
        self.assertEqual(
            copied_middle.position[0] - copied_source.position[0],
            middle_node.position[0] - workflow.get_node(source_id).position[0],
        )

        internal_copied_connections = [
            conn for conn in workflow.connections
            if conn.source_node_id in new_nodes and conn.target_node_id in new_nodes
        ]
        copied_external_connections = [
            conn for conn in workflow.connections
            if (conn.source_node_id in new_nodes) ^ (conn.target_node_id in new_nodes)
        ]
        self.assertEqual(len(internal_copied_connections), 1)
        self.assertEqual(len(copied_external_connections), 0)

    def test_model_copy_paste_preserves_parameters_and_internal_connections_only(self):
        widget = ModelBuilderView()
        self.addCleanup(widget.close)
        widget.resize(900, 600)
        widget.show()

        graph_widget = widget._graph_widget
        model_graph = widget.get_model_graph()

        input_id = graph_widget.add_layer("input", (20.0, 40.0))
        dense_id = graph_widget.add_layer("dense", (200.0, 40.0))
        output_id = graph_widget.add_layer("output", (380.0, 40.0))

        self.assertIsNotNone(input_id)
        self.assertIsNotNone(dense_id)
        self.assertIsNotNone(output_id)

        success, _ = model_graph.connect(input_id, dense_id)
        self.assertTrue(success)
        graph_widget._scene.add_connection_item(input_id, dense_id)

        success, _ = model_graph.connect(dense_id, output_id)
        self.assertTrue(success)
        graph_widget._scene.add_connection_item(dense_id, output_id)

        dense_layer = model_graph.get_layer(dense_id)
        param_name, param_value = _choose_distinct_parameter_value(dense_layer)
        success, message = dense_layer.set_parameter(param_name, param_value)
        self.assertTrue(success, message)

        graph_widget._scene.clearSelection()
        graph_widget._scene.layer_items[input_id].setSelected(True)
        graph_widget._scene.layer_items[dense_id].setSelected(True)
        self.app.processEvents()

        self.assertTrue(graph_widget._copy_selected())
        self.assertTrue(graph_widget._paste_clipboard())

        self.assertEqual(len(model_graph.layers), 5)
        self.assertEqual(len(model_graph.connections), 3)

        original_ids = {input_id, dense_id, output_id}
        new_layers = {
            layer_id: layer
            for layer_id, layer in model_graph.layers.items()
            if layer_id not in original_ids
        }
        self.assertEqual({layer.layer_type for layer in new_layers.values()}, {"input", "dense"})

        copied_dense = next(layer for layer in new_layers.values() if layer.layer_type == "dense")
        copied_input = next(layer for layer in new_layers.values() if layer.layer_type == "input")

        self.assertEqual(copied_dense.get_parameter(param_name), param_value)
        self.assertNotEqual(copied_dense.layer_id, dense_id)
        self.assertNotEqual(copied_input.layer_id, input_id)
        self.assertNotEqual(copied_dense.position, dense_layer.position)
        self.assertEqual(
            copied_dense.position[0] - copied_input.position[0],
            dense_layer.position[0] - model_graph.get_layer(input_id).position[0],
        )

        internal_copied_connections = [
            conn for conn in model_graph.connections
            if conn.source_layer_id in new_layers and conn.target_layer_id in new_layers
        ]
        copied_external_connections = [
            conn for conn in model_graph.connections
            if (conn.source_layer_id in new_layers) ^ (conn.target_layer_id in new_layers)
        ]
        self.assertEqual(len(internal_copied_connections), 1)
        self.assertEqual(len(copied_external_connections), 0)
