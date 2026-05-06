import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QGraphicsTextItem

from src.model_builder import get_all_layer_types
from src.ui.views.model_builder_view import ModelBuilderView
from src.ui.views.workflow_editor_widget import WorkflowEditorWidget
from src.workflow import Workflow
from src.workflow.node_base import get_all_node_types


class DoubleClickCenterPlacementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_workflow_double_click_add_uses_visible_viewport_center(self):
        widget = WorkflowEditorWidget()
        self.addCleanup(widget.close)
        widget.resize(900, 600)
        widget.show()

        workflow = Workflow("test_workflow")
        widget.set_workflow(workflow)

        node_graph = widget._node_graph
        node_graph._scene.setSceneRect(-2000, -2000, 4000, 4000)
        node_graph._view.scale(1.5, 1.5)
        node_graph._view.centerOn(420, 260)
        self.app.processEvents()

        expected_center = node_graph._view.mapToScene(
            node_graph._view.viewport().rect().center()
        )

        widget._on_add_node_from_palette(get_all_node_types()[0])

        self.assertEqual(len(workflow.nodes), 1)
        node = next(iter(workflow.nodes.values()))
        self.assertAlmostEqual(node.position[0], expected_center.x(), delta=2.0)
        self.assertAlmostEqual(node.position[1], expected_center.y(), delta=2.0)

    def test_model_double_click_add_uses_visible_viewport_center(self):
        widget = ModelBuilderView()
        self.addCleanup(widget.close)
        widget.resize(900, 600)
        widget.show()

        graph_widget = widget._graph_widget
        graph_widget._scene.setSceneRect(-2000, -2000, 4000, 4000)
        graph_widget._view.scale(1.25, 1.25)
        graph_widget._view.centerOn(360, 180)
        self.app.processEvents()

        expected_center = graph_widget._view.mapToScene(
            graph_widget._view.viewport().rect().center()
        )

        widget._on_add_layer(get_all_layer_types()[0])

        self.assertIsNotNone(widget.get_model_graph())
        self.assertEqual(len(widget.get_model_graph().layers), 1)
        layer = next(iter(widget.get_model_graph().layers.values()))
        self.assertAlmostEqual(layer.position[0], expected_center.x(), delta=2.0)
        self.assertAlmostEqual(layer.position[1], expected_center.y(), delta=2.0)

    def test_workflow_node_text_items_remain_visible_after_creation(self):
        widget = WorkflowEditorWidget()
        self.addCleanup(widget.close)

        workflow = Workflow("text_visibility_workflow")
        widget.set_workflow(workflow)

        node_id = widget._node_graph.add_node(get_all_node_types()[0], (0.0, 0.0))
        node_item = widget._node_graph._scene.node_items[node_id]

        text_items = [
            item for item in node_item.childItems()
            if isinstance(item, QGraphicsTextItem)
        ]

        self.assertGreaterEqual(len(text_items), 1)
        self.assertTrue(any(item.toPlainText().strip() for item in text_items))

    def test_model_layer_text_items_remain_visible_after_creation(self):
        widget = ModelBuilderView()
        self.addCleanup(widget.close)

        layer_id = widget._graph_widget.add_layer(get_all_layer_types()[0], (0.0, 0.0))
        layer_item = widget._graph_widget._scene.layer_items[layer_id]

        text_items = [
            item for item in layer_item.childItems()
            if isinstance(item, QGraphicsTextItem)
        ]

        self.assertGreaterEqual(len(text_items), 1)
        self.assertTrue(any(item.toPlainText().strip() for item in text_items))
