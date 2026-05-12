import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest

from src.model_builder.model_graph import ModelGraph
from src.ui.views.model_builder_view import ModelBuilderView
from src.ui.views.workflow_editor_widget import WorkflowEditorWidget
from src.workflow import Workflow


class CanvasFitOriginTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_workflow_fit_recenters_content_coordinates_around_origin(self):
        widget = WorkflowEditorWidget()
        self.addCleanup(widget.close)
        widget.resize(900, 600)
        widget.show()

        workflow = Workflow("fit_origin_workflow")
        widget.set_workflow(workflow)
        node_graph = widget._node_graph

        source_id = node_graph.add_node("audio_file", (2400.0, -1200.0))
        sink_id = node_graph.add_node("fft", (2750.0, -960.0))
        self.assertIsNotNone(source_id)
        self.assertIsNotNone(sink_id)

        before_dx = workflow.get_node(sink_id).position[0] - workflow.get_node(source_id).position[0]
        before_dy = workflow.get_node(sink_id).position[1] - workflow.get_node(source_id).position[1]

        node_graph.fit_to_selection()
        self.app.processEvents()

        after_center = node_graph._scene.itemsBoundingRect().center()
        viewport_center = node_graph._view.mapToScene(node_graph._view.viewport().rect().center())

        self.assertAlmostEqual(after_center.x(), 0.0, delta=2.0)
        self.assertAlmostEqual(after_center.y(), 0.0, delta=2.0)
        self.assertAlmostEqual(viewport_center.x(), 0.0, delta=3.0)
        self.assertAlmostEqual(viewport_center.y(), 0.0, delta=3.0)
        self.assertAlmostEqual(
            workflow.get_node(sink_id).position[0] - workflow.get_node(source_id).position[0],
            before_dx,
            delta=0.01,
        )
        self.assertAlmostEqual(
            workflow.get_node(sink_id).position[1] - workflow.get_node(source_id).position[1],
            before_dy,
            delta=0.01,
        )

    def test_model_fit_recenters_content_coordinates_around_origin(self):
        widget = ModelBuilderView()
        self.addCleanup(widget.close)
        widget.resize(900, 600)
        widget.show()

        graph_widget = widget._graph_widget
        model_graph = widget.get_model_graph()

        input_id = graph_widget.add_layer("input", (-3100.0, 1800.0))
        output_id = graph_widget.add_layer("output", (-2800.0, 1960.0))
        self.assertIsNotNone(input_id)
        self.assertIsNotNone(output_id)

        before_dx = model_graph.get_layer(output_id).position[0] - model_graph.get_layer(input_id).position[0]
        before_dy = model_graph.get_layer(output_id).position[1] - model_graph.get_layer(input_id).position[1]

        graph_widget.fit_to_selection()
        self.app.processEvents()

        after_center = graph_widget._scene.itemsBoundingRect().center()
        viewport_center = graph_widget._view.mapToScene(graph_widget._view.viewport().rect().center())

        self.assertAlmostEqual(after_center.x(), 0.0, delta=2.0)
        self.assertAlmostEqual(after_center.y(), 0.0, delta=2.0)
        self.assertAlmostEqual(viewport_center.x(), 0.0, delta=3.0)
        self.assertAlmostEqual(viewport_center.y(), 0.0, delta=3.0)
        self.assertAlmostEqual(
            model_graph.get_layer(output_id).position[0] - model_graph.get_layer(input_id).position[0],
            before_dx,
            delta=0.01,
        )
        self.assertAlmostEqual(
            model_graph.get_layer(output_id).position[1] - model_graph.get_layer(input_id).position[1],
            before_dy,
            delta=0.01,
        )

    def test_empty_workflow_fit_restores_default_origin(self):
        widget = WorkflowEditorWidget()
        self.addCleanup(widget.close)
        widget.resize(900, 600)
        widget.show()

        workflow = Workflow("empty_fit_origin_workflow")
        widget.set_workflow(workflow)
        node_graph = widget._node_graph
        node_graph._view.scale(1.5, 1.5)
        node_graph._view.centerOn(1800.0, -900.0)

        node_graph.fit_to_selection()
        self.app.processEvents()

        viewport_center = node_graph._view.mapToScene(node_graph._view.viewport().rect().center())
        self.assertAlmostEqual(viewport_center.x(), 0.0, delta=2.0)
        self.assertAlmostEqual(viewport_center.y(), 0.0, delta=2.0)
        self.assertAlmostEqual(node_graph._view.transform().m11(), 1.0, delta=0.01)

    def test_workflow_canvas_auto_fits_after_initial_workflow_load(self):
        widget = WorkflowEditorWidget()
        self.addCleanup(widget.close)
        widget.resize(900, 600)
        widget.show()

        workflow = Workflow("startup_fit_workflow")
        workflow.create_and_add_node("audio_file", (2200.0, 1400.0))
        workflow.create_and_add_node("fft", (2500.0, 1600.0))

        widget.set_workflow(workflow)
        self.app.processEvents()

        node_graph = widget._node_graph
        after_center = node_graph._scene.itemsBoundingRect().center()
        viewport_center = node_graph._view.mapToScene(node_graph._view.viewport().rect().center())

        self.assertAlmostEqual(after_center.x(), 0.0, delta=2.0)
        self.assertAlmostEqual(after_center.y(), 0.0, delta=2.0)
        self.assertAlmostEqual(viewport_center.x(), 0.0, delta=3.0)
        self.assertAlmostEqual(viewport_center.y(), 0.0, delta=3.0)

    def test_model_canvas_auto_fits_after_initial_graph_load(self):
        widget = ModelBuilderView()
        self.addCleanup(widget.close)
        widget.resize(900, 600)
        widget.show()

        model_graph = ModelGraph("startup_fit_model")
        model_graph.create_and_add_layer("input", (-2500.0, -1500.0))
        model_graph.create_and_add_layer("output", (-2200.0, -1360.0))

        widget.set_model_graph(model_graph)
        self.app.processEvents()

        graph_widget = widget._graph_widget
        after_center = graph_widget._scene.itemsBoundingRect().center()
        viewport_center = graph_widget._view.mapToScene(graph_widget._view.viewport().rect().center())

        self.assertAlmostEqual(after_center.x(), 0.0, delta=2.0)
        self.assertAlmostEqual(after_center.y(), 0.0, delta=2.0)
        self.assertAlmostEqual(viewport_center.x(), 0.0, delta=3.0)
        self.assertAlmostEqual(viewport_center.y(), 0.0, delta=3.0)

    def test_workflow_auto_fit_waits_until_canvas_is_visible_and_sized(self):
        widget = WorkflowEditorWidget()
        self.addCleanup(widget.close)

        workflow = Workflow("hidden_startup_fit_workflow")
        workflow.create_and_add_node("audio_file", (-700.0, -300.0))
        workflow.create_and_add_node("fft", (500.0, 200.0))

        widget.set_workflow(workflow)
        self.app.processEvents()

        widget.resize(900, 600)
        widget.show()
        QTest.qWait(80)

        node_graph = widget._node_graph
        viewport_center = node_graph._view.mapToScene(node_graph._view.viewport().rect().center())

        self.assertGreater(node_graph._view.transform().m11(), 0.2)
        self.assertAlmostEqual(viewport_center.x(), 0.0, delta=5.0)
        self.assertAlmostEqual(viewport_center.y(), 0.0, delta=5.0)

    def test_model_auto_fit_waits_until_canvas_is_visible_and_sized(self):
        widget = ModelBuilderView()
        self.addCleanup(widget.close)

        model_graph = ModelGraph("hidden_startup_fit_model")
        model_graph.create_and_add_layer("input", (-700.0, -300.0))
        model_graph.create_and_add_layer("output", (500.0, 200.0))

        widget.set_model_graph(model_graph)
        self.app.processEvents()

        widget.resize(900, 600)
        widget.show()
        QTest.qWait(80)

        graph_widget = widget._graph_widget
        viewport_center = graph_widget._view.mapToScene(graph_widget._view.viewport().rect().center())

        self.assertGreater(graph_widget._view.transform().m11(), 0.2)
        self.assertAlmostEqual(viewport_center.x(), 0.0, delta=5.0)
        self.assertAlmostEqual(viewport_center.y(), 0.0, delta=5.0)


if __name__ == "__main__":
    unittest.main()
