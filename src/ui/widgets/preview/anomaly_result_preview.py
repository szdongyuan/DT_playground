# -*- coding: utf-8 -*-
"""Interactive preview for thresholded anomaly-detection results."""

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.ui.i18n import tr_
from src.ui.styles import Styles
from src.workflow.nodes.anomaly import AnomalyResultData

from .audio_preview import AudioPreviewWidget
from .base_preview import BasePreviewWidget, register_preview
from .feature_1d_preview import Feature1DPreviewWidget
from .feature_2d_preview import Feature2DPreviewWidget


logger = logging.getLogger(__name__)


try:
    import pyqtgraph as pg

    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False


@register_preview
class AnomalyResultPreviewWidget(BasePreviewWidget):
    """Show anomaly summary, distribution, ranking, and selected source data."""

    display_name = tr_("Anomaly explorer")
    icon = "🚨"

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._result: Optional[AnomalyResultData] = None
        self._row_indices: List[int] = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setHandleWidth(2)
        layout.addWidget(splitter)

        overview = QWidget()
        overview_layout = QHBoxLayout(overview)
        overview_layout.setContentsMargins(0, 0, 0, 0)

        summary_group = QGroupBox(tr_("Anomaly summary"))
        summary_group.setStyleSheet(Styles.group_box(Styles.COLORS["red"]))
        summary_layout = QVBoxLayout(summary_group)
        self._summary_label = QLabel()
        self._summary_label.setWordWrap(True)
        self._summary_label.setStyleSheet(
            f"color: {Styles.COLORS['text']}; font-family: 'Consolas', monospace;"
        )
        summary_layout.addWidget(self._summary_label)
        overview_layout.addWidget(summary_group, 1)

        distribution_group = QGroupBox(tr_("Score distribution"))
        distribution_group.setStyleSheet(Styles.group_box(Styles.COLORS["lavender"]))
        distribution_layout = QVBoxLayout(distribution_group)
        if HAS_PYQTGRAPH:
            self._plot_widget = pg.PlotWidget()
            self._plot_widget.setBackground(Styles.COLORS["mantle"])
            self._plot_widget.setLabel("bottom", tr_("Normalized anomaly score"))
            self._plot_widget.setLabel("left", tr_("Count"))
            distribution_layout.addWidget(self._plot_widget)
        else:
            self._plot_widget = None
            placeholder = QLabel(tr_("Please install pyqtgraph to display charts"))
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            distribution_layout.addWidget(placeholder)
        overview_layout.addWidget(distribution_group, 2)
        splitter.addWidget(overview)

        table_group = QGroupBox(tr_("Ranked samples"))
        table_group.setStyleSheet(Styles.group_box(Styles.COLORS["blue"]))
        table_layout = QVBoxLayout(table_group)
        self._table = QTableWidget()
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels(
            [tr_("Rank"), tr_("Sample"), tr_("Raw score"), tr_("Score (0-100)"), tr_("Severity")]
        )
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.currentCellChanged.connect(self._show_source_for_row)
        table_layout.addWidget(self._table)
        splitter.addWidget(table_group)

        detail_group = QGroupBox(tr_("Selected sample preview"))
        detail_group.setStyleSheet(Styles.group_box(Styles.COLORS["green"]))
        detail_layout = QVBoxLayout(detail_group)
        self._detail_stack = QStackedWidget()
        self._detail_empty = QLabel(tr_("Select a sample to preview its source data"))
        self._detail_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._audio_preview = AudioPreviewWidget()
        self._feature_1d_preview = Feature1DPreviewWidget()
        self._feature_2d_preview = Feature2DPreviewWidget()
        for widget in (
            self._detail_empty,
            self._audio_preview,
            self._feature_1d_preview,
            self._feature_2d_preview,
        ):
            self._detail_stack.addWidget(widget)
        detail_layout.addWidget(self._detail_stack)
        splitter.addWidget(detail_group)
        splitter.setSizes([220, 280, 360])

    @classmethod
    def can_display(cls, data: Any) -> bool:
        return isinstance(data, AnomalyResultData)

    def set_data(self, data: Any) -> bool:
        if not self.can_display(data):
            self.clear()
            return False

        self._current_data = data
        self._result = data
        scores = data.scores.normalized_scores
        anomaly_count = int(np.count_nonzero(data.is_anomaly))
        attention_count = data.severities.count("attention")
        normal_count = data.severities.count("normal")
        self._update_data_info(
            f"🚨 ANOMALY_RESULT  samples={len(scores)}  anomalies={anomaly_count}  "
            f"threshold={data.threshold:.2f}"
        )
        self._summary_label.setText(
            tr_(
                "Samples: {samples}\nAnomalies: {anomalies}\nAttention: {attention}\n"
                "Normal: {normal}\nThreshold: {threshold:.2f}\nStrategy: {strategy}"
            ).format(
                samples=len(scores),
                anomalies=anomaly_count,
                attention=attention_count,
                normal=normal_count,
                threshold=data.threshold,
                strategy=data.strategy,
            )
        )
        self._update_plot(scores, data.threshold, data.attention_threshold)
        self._update_table(data)
        self.data_changed.emit()
        return True

    def _update_plot(self, scores: np.ndarray, threshold: float, attention_threshold: float):
        if self._plot_widget is None:
            return
        self._plot_widget.clear()
        counts, edges = np.histogram(scores, bins=min(20, max(5, len(scores))))
        centers = (edges[:-1] + edges[1:]) / 2.0
        widths = np.diff(edges) * 0.9
        self._plot_widget.addItem(
            pg.BarGraphItem(
                x=centers,
                height=counts,
                width=widths,
                brush=Styles.COLORS["blue"],
            )
        )
        self._plot_widget.addItem(
            pg.InfiniteLine(pos=attention_threshold, angle=90, pen=Styles.COLORS["yellow"])
        )
        self._plot_widget.addItem(
            pg.InfiniteLine(pos=threshold, angle=90, pen=Styles.COLORS["red"])
        )

    def _update_table(self, result: AnomalyResultData):
        normalized = result.scores.normalized_scores
        severity_labels = {
            "normal": tr_("Normal"),
            "attention": tr_("Attention"),
            "anomaly": tr_("Anomaly"),
        }
        self._row_indices = np.argsort(normalized)[::-1].astype(int).tolist()
        self._table.blockSignals(True)
        self._table.setRowCount(len(self._row_indices))
        for row, sample_index in enumerate(self._row_indices):
            sample_id = result.scores.sample_ids[sample_index]
            display_id = os.path.basename(sample_id) or sample_id
            values = [
                str(row + 1),
                display_id,
                f"{result.scores.raw_scores[sample_index]:.6f}",
                f"{normalized[sample_index]:.2f}",
                severity_labels.get(result.severities[sample_index], result.severities[sample_index]),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, sample_index)
                self._table.setItem(row, column, item)
        self._table.blockSignals(False)
        if self._row_indices:
            self._table.selectRow(0)
            self._show_source_for_row(0)

    def _show_source_for_row(self, row: int, *_args):
        if self._result is None or row < 0 or row >= len(self._row_indices):
            self._detail_stack.setCurrentWidget(self._detail_empty)
            return
        source_items = self._result.scores.source_items
        sample_index = self._row_indices[row]
        if not source_items or sample_index >= len(source_items):
            self._detail_stack.setCurrentWidget(self._detail_empty)
            return

        source = source_items[sample_index]
        if AudioPreviewWidget.can_display(source) and self._audio_preview.set_data(source):
            self._detail_stack.setCurrentWidget(self._audio_preview)
        elif Feature1DPreviewWidget.can_display(source) and self._feature_1d_preview.set_data(source):
            self._detail_stack.setCurrentWidget(self._feature_1d_preview)
        elif Feature2DPreviewWidget.can_display(source) and self._feature_2d_preview.set_data(source):
            self._detail_stack.setCurrentWidget(self._feature_2d_preview)
        else:
            self._detail_stack.setCurrentWidget(self._detail_empty)

    def clear(self):
        self._current_data = None
        self._result = None
        self._row_indices = []
        self._data_info = ""
        self._summary_label.clear()
        self._table.setRowCount(0)
        if self._plot_widget is not None:
            self._plot_widget.clear()
        self._audio_preview.clear()
        self._feature_1d_preview.clear()
        self._feature_2d_preview.clear()
        self._detail_stack.setCurrentWidget(self._detail_empty)
