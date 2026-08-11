# -*- coding: utf-8 -*-
"""Preview widget for all line-compatible records in one chart."""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.ui.i18n import tr_
from src.ui.styles import Styles
from src.workflow.nodes.visualization import MultiCurvePreviewData

from .base_preview import BasePreviewWidget, register_preview

try:
    import pyqtgraph as pg

    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False


_SERIES_COLORS = [
    "#89b4fa",
    "#cba6f7",
    "#94e2d5",
    "#fab387",
    "#f38ba8",
    "#a6e3a1",
    "#f5c2e7",
    "#74c7ec",
    "#b4befe",
    "#eba0ac",
]


@register_preview
class MultiCurvePreviewWidget(BasePreviewWidget):
    """Render curves in either batched or individually selectable mode."""

    display_name = tr_("Multi-curve viewer")
    icon = "📉"

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._curve_items: List[Any] = []
        self._series_curve_items: List[Any] = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        self._series_panel = QWidget()
        self._series_panel.setStyleSheet(f"""
            QWidget {{
                background: {Styles.COLORS['base']};
                color: {Styles.COLORS['text']};
            }}
            QLineEdit {{
                background: {Styles.COLORS['surface0']};
                color: {Styles.COLORS['text']};
                border: 1px solid {Styles.COLORS['surface1']};
                border-radius: 4px;
                padding: 6px;
            }}
            QListWidget {{
                background: {Styles.COLORS['mantle']};
                color: {Styles.COLORS['text']};
                border: 1px solid {Styles.COLORS['surface1']};
                border-radius: 4px;
            }}
            QListWidget::item {{
                padding: 3px;
            }}
            QListWidget::item:hover {{
                background: {Styles.COLORS['surface0']};
            }}
        """)
        series_layout = QVBoxLayout(self._series_panel)
        series_layout.setContentsMargins(0, 0, 4, 0)
        series_layout.addWidget(QLabel(tr_("Data series")))

        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText(tr_("Search series..."))
        self._search_edit.textChanged.connect(self._filter_series)
        self._search_edit.setStyleSheet(Styles.FORM_CONTROLS)
        series_layout.addWidget(self._search_edit)

        self._series_list = QListWidget()
        self._series_list.setStyleSheet(Styles.TREE_WIDGET)
        self._series_list.itemChanged.connect(self._on_series_item_changed)
        series_layout.addWidget(self._series_list, 1)
        self._series_panel.setMinimumWidth(210)
        self._series_panel.setMaximumWidth(340)
        splitter.addWidget(self._series_panel)

        if HAS_PYQTGRAPH:
            self._plot_widget = pg.PlotWidget()
            self._plot_widget.setBackground(Styles.COLORS["mantle"])
            self._plot_widget.showGrid(x=True, y=True, alpha=0.25)
            self._plot_widget.setLabel("left", tr_("Value"))
            font = QFont("Microsoft YaHei", 9)
            for axis_name in ("left", "bottom"):
                axis = self._plot_widget.getPlotItem().getAxis(axis_name)
                axis.setTickFont(font)
                axis.setStyle(tickTextOffset=5)
            splitter.addWidget(self._plot_widget)
        else:
            self._plot_widget = None
            placeholder = QLabel(tr_("Please install pyqtgraph to display charts"))
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            splitter.addWidget(placeholder)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)

        self._status_label = QLabel()
        self._status_label.setStyleSheet(
            f"color: {Styles.COLORS['subtext1']}; padding: 4px;"
        )
        layout.addWidget(self._status_label)

    def set_data(self, data: Any) -> bool:
        if not isinstance(data, MultiCurvePreviewData):
            return False

        self._current_data = data
        selectable = data.interaction_mode == "series_toggle"
        self._series_panel.setVisible(selectable)
        self._populate_series_list() if selectable else self._clear_series_list()

        self._update_data_info(
            tr_("MULTI_CURVE  curves={curves}").format(curves=len(data.series))
        )
        self._draw_plot()
        self.data_changed.emit()
        return True

    def _clear_series_list(self):
        self._series_list.blockSignals(True)
        self._series_list.clear()
        self._series_list.blockSignals(False)

    def _populate_series_list(self):
        self._series_list.blockSignals(True)
        self._series_list.clear()
        for index, series in enumerate(self._current_data.series):
            item = QListWidgetItem(series.name)
            item.setData(Qt.ItemDataRole.UserRole, index)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            if HAS_PYQTGRAPH:
                item.setForeground(
                    pg.mkColor(_SERIES_COLORS[index % len(_SERIES_COLORS)])
                )
            self._series_list.addItem(item)
        self._series_list.blockSignals(False)

    def _offset_step(self) -> float:
        if not isinstance(self._current_data, MultiCurvePreviewData):
            return 0.0
        minimum = min(float(np.min(series.y)) for series in self._current_data.series)
        maximum = max(float(np.max(series.y)) for series in self._current_data.series)
        span = maximum - minimum
        return span * 0.35 if span > 0 else 1.0

    def _draw_plot(self):
        if not HAS_PYQTGRAPH or self._plot_widget is None:
            return
        if not isinstance(self._current_data, MultiCurvePreviewData):
            return

        data = self._current_data
        plot_item = self._plot_widget.getPlotItem()
        plot_item.clear()
        if plot_item.legend is not None:
            plot_item.legend.scene().removeItem(plot_item.legend)
            plot_item.legend = None

        self._curve_items = []
        self._series_curve_items = []
        offset_step = self._offset_step() if data.display_mode == "offset" else 0.0
        if data.interaction_mode == "series_toggle":
            self._draw_selectable_curves(plot_item, offset_step)
        else:
            self._draw_batched_curves(plot_item, offset_step)

        self._draw_statistics(plot_item)
        plot_item.setLabel("bottom", data.x_label)
        plot_item.setLabel("left", tr_("Value"))
        self._update_status(len(data.series))

    def _draw_selectable_curves(self, plot_item: Any, offset_step: float):
        data = self._current_data
        if data.show_legend:
            plot_item.addLegend(offset=(10, 10))
        opacity = int(round(data.line_opacity * 255))
        for series_index, series in enumerate(data.series):
            color = pg.mkColor(_SERIES_COLORS[series_index % len(_SERIES_COLORS)])
            color.setAlpha(opacity)
            curve = plot_item.plot(
                series.x,
                series.y + series_index * offset_step,
                pen=pg.mkPen(color=color, width=1.2),
                name=series.name if data.show_legend else None,
            )
            self._curve_items.append(curve)
            self._series_curve_items.append(curve)

    def _draw_batched_curves(self, plot_item: Any, offset_step: float):
        data = self._current_data
        opacity = int(round(data.line_opacity * 255))
        color_count = min(len(_SERIES_COLORS), len(data.series))
        for color_index in range(color_count):
            indices = range(color_index, len(data.series), len(_SERIES_COLORS))
            group = [(index, data.series[index]) for index in indices]
            point_count = sum(series.y.size + 1 for _, series in group)
            x_values = np.empty(point_count, dtype=np.float32)
            y_values = np.empty(point_count, dtype=np.float32)
            cursor = 0
            for series_index, series in group:
                next_cursor = cursor + series.y.size
                x_values[cursor:next_cursor] = series.x
                y_values[cursor:next_cursor] = (
                    series.y + np.float32(series_index * offset_step)
                )
                x_values[next_cursor] = np.nan
                y_values[next_cursor] = np.nan
                cursor = next_cursor + 1

            color = pg.mkColor(_SERIES_COLORS[color_index])
            color.setAlpha(opacity)
            curve = plot_item.plot(
                x_values,
                y_values,
                connect="finite",
                pen=pg.mkPen(color=color, width=1.2),
            )
            self._curve_items.append(curve)

    def _draw_statistics(self, plot_item: Any):
        data = self._current_data
        if (
            data.aggregate_x is not None
            and data.lower_y is not None
            and data.upper_y is not None
        ):
            lower = pg.PlotDataItem(data.aggregate_x, data.lower_y)
            upper = pg.PlotDataItem(data.aggregate_x, data.upper_y)
            plot_item.addItem(lower)
            plot_item.addItem(upper)
            band_color = pg.mkColor(Styles.COLORS["yellow"])
            band_color.setAlpha(45)
            plot_item.addItem(
                pg.FillBetweenItem(lower, upper, brush=band_color)
            )

        if data.aggregate_x is not None and data.aggregate_y is not None:
            plot_item.plot(
                data.aggregate_x,
                data.aggregate_y,
                pen=pg.mkPen(Styles.COLORS["yellow"], width=3),
                name=(
                    tr_("Aggregate")
                    if data.show_legend
                    and data.interaction_mode == "series_toggle"
                    else None
                ),
            )

    def _on_series_item_changed(self, item: QListWidgetItem):
        if not isinstance(self._current_data, MultiCurvePreviewData):
            return
        series_index = int(item.data(Qt.ItemDataRole.UserRole))
        if series_index >= len(self._series_curve_items):
            return
        self._series_curve_items[series_index].setVisible(
            item.checkState() == Qt.CheckState.Checked
        )
        visible_count = sum(
            self._series_list.item(row).checkState() == Qt.CheckState.Checked
            for row in range(self._series_list.count())
        )
        self._update_status(visible_count)

    def _update_status(self, visible_count: int):
        if not isinstance(self._current_data, MultiCurvePreviewData):
            self._status_label.clear()
            return
        data = self._current_data
        if data.interaction_mode == "series_toggle":
            summary = tr_("{visible} / {total} curves visible").format(
                visible=visible_count,
                total=len(data.series),
            )
        else:
            summary = tr_("{total} curves").format(total=len(data.series))
        parts = [summary]
        if data.downsampled:
            parts.append(tr_("display downsampling active"))
        if data.warnings:
            parts.append(data.warnings[0])
            if len(data.warnings) > 1:
                parts.append(
                    tr_("{count} more warning(s)").format(
                        count=len(data.warnings) - 1
                    )
                )
        self._status_label.setText(" · ".join(parts))
        self._status_label.setToolTip("\n".join(data.warnings))

    def _filter_series(self, text: str):
        needle = text.strip().casefold()
        for row in range(self._series_list.count()):
            item = self._series_list.item(row)
            item.setHidden(bool(needle) and needle not in item.text().casefold())

    def clear(self):
        self._current_data = None
        self._data_info = ""
        self._clear_series_list()
        self._series_panel.hide()
        self._status_label.clear()
        self._curve_items = []
        self._series_curve_items = []
        if HAS_PYQTGRAPH and self._plot_widget is not None:
            self._plot_widget.clear()

    @classmethod
    def can_display(cls, data: Any) -> bool:
        return isinstance(data, MultiCurvePreviewData)
