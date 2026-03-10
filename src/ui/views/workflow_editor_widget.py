# -*- coding: utf-8 -*-
"""
Workflow Editor Widget

Pure editor core: Splitter + NodePalette + NodeGraphWidget + PropertyPanel.
Contains no toolbar, no file dialogs, no run/stop logic, no config persistence.
"""

import logging
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QSplitter, QVBoxLayout, QWidget

from ..node_editor import NodeGraphWidget, NodePalette, PropertyPanel
from ..styles import Styles
from ...workflow.workflow import Workflow


logger = logging.getLogger(__name__)


class WorkflowEditorWidget(QWidget):
    """
    Pure workflow editor widget.

    Provides the three-panel editor layout (palette | graph | properties)
    and exposes editing signals / node-state APIs.
    """

    workflow_changed = pyqtSignal()
    node_selected = pyqtSignal(str)
    node_double_clicked = pyqtSignal(str)
    run_from_node_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._workflow: Optional[Workflow] = None
        self._setup_ui()
        self._connect_signals()

    # ===== UI =====

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)
        splitter.setStyleSheet(f"""
            QSplitter::handle {{
                background: {Styles.COLORS['surface1']};
            }}
        """)
        layout.addWidget(splitter)

        self._node_palette = NodePalette()
        self._node_palette.setMinimumWidth(180)
        self._node_palette.setMaximumWidth(280)
        splitter.addWidget(self._node_palette)

        self._node_graph = NodeGraphWidget()
        splitter.addWidget(self._node_graph)

        self._property_panel = PropertyPanel()
        self._property_panel.setMinimumWidth(250)
        self._property_panel.setMaximumWidth(350)
        splitter.addWidget(self._property_panel)

        splitter.setSizes([200, 600, 280])

    def _connect_signals(self):
        self._node_palette.node_add_requested.connect(self._on_add_node_from_palette)
        self._node_graph.node_selected.connect(self._on_node_selected)
        self._node_graph.node_double_clicked.connect(self.node_double_clicked)
        self._node_graph.run_from_node_requested.connect(self.run_from_node_requested)
        self._node_graph.workflow_changed.connect(self._on_workflow_changed)
        self._property_panel.parameter_changed.connect(self._on_parameter_changed)

    # ===== Workflow data =====

    def set_workflow(self, workflow: Workflow):
        self._workflow = workflow
        self._node_graph.set_workflow(workflow)
        self._property_panel.clear()

    def get_workflow(self) -> Optional[Workflow]:
        return self._node_graph.get_workflow()

    # ===== Node execution state =====

    def update_node_state(self, node_id: str, state: str):
        self._node_graph.update_node_state(node_id, state)

    def reset_all_node_states(self):
        self._node_graph.reset_all_node_states()

    def reset_node_states(self, node_ids: list[str]):
        self._node_graph.reset_node_states(node_ids)

    def highlight_node(self, node_id: str):
        self._node_graph.highlight_node(node_id)

    def fit_to_selection(self):
        self._node_graph.fit_to_selection()

    # ===== Internal event handlers =====

    def _on_add_node_from_palette(self, node_type: str):
        self._node_graph.add_node(node_type, (100, 100))

    def _on_node_selected(self, node_id: str):
        workflow = self._node_graph.get_workflow()
        if workflow and node_id:
            node = workflow.get_node(node_id)
            if node:
                self._property_panel.set_node(node, node_id)
                self.node_selected.emit(node_id)

    def _on_workflow_changed(self):
        self.workflow_changed.emit()

    def _on_parameter_changed(self, node_id: str, param_name: str, value):
        workflow = self._node_graph.get_workflow()
        if workflow:
            workflow._mark_dirty()
        self.workflow_changed.emit()
