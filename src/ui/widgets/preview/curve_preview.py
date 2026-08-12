# -*- coding: utf-8 -*-
"""Preview one physical-coordinate curve record and one channel at a time."""

from typing import Any, Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QGroupBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from src.ui.i18n import tr_
from src.ui.styles import Styles
from src.workflow.curve import CurveData

from .base_preview import BasePreviewWidget, register_preview

try:
    import pyqtgraph as pg
    from PySide6.QtGui import QFont

    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False


@register_preview
class CurvePreviewWidget(BasePreviewWidget):
    """Render one channel selected from a single CurveData record."""

    display_name = tr_("Physical curve")
    icon = "📈"

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        curve_group = QGroupBox(tr_("Physical curve"))
        curve_group.setStyleSheet(Styles.group_box(Styles.COLORS["teal"]))
        curve_layout = QVBoxLayout(curve_group)
        curve_layout.setContentsMargins(4, 4, 4, 4)

        channel_row = QHBoxLayout()
        channel_row.addWidget(QLabel(tr_("Channel:")))
        self._channel_combo = QComboBox()
        self._channel_combo.setStyleSheet(Styles.FORM_CONTROLS)
        self._channel_combo.currentIndexChanged.connect(self._on_channel_changed)
        channel_row.addWidget(self._channel_combo)
        channel_row.addStretch()
        curve_layout.addLayout(channel_row)

        if HAS_PYQTGRAPH:
            self._plot_widget = pg.PlotWidget()
            self._plot_widget.setBackground(Styles.COLORS["mantle"])
            self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
            font = QFont("Microsoft YaHei", 9)
            for axis_name in ("left", "bottom"):
                axis = self._plot_widget.getPlotItem().getAxis(axis_name)
                axis.setTickFont(font)
                axis.setStyle(tickTextOffset=5)
            self._curve = self._plot_widget.plot(
                pen=pg.mkPen(color=Styles.COLORS["teal"], width=2)
            )
            self._plot_widget.setMinimumHeight(400)
            curve_layout.addWidget(self._plot_widget)
        else:
            self._plot_widget = None
            placeholder = QLabel(tr_("Please install pyqtgraph to display charts"))
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            curve_layout.addWidget(placeholder)
        layout.addWidget(curve_group)

        stats_group = QGroupBox(tr_("Statistics"))
        stats_group.setStyleSheet(Styles.group_box(Styles.COLORS["lavender"]))
        stats_layout = QVBoxLayout(stats_group)
        self._stats_label = QLabel()
        self._stats_label.setStyleSheet(
            f"color: {Styles.COLORS['text']}; background: {Styles.COLORS['mantle']}; "
            "font-family: 'Consolas', 'Courier New', monospace; padding: 8px;"
        )
        stats_layout.addWidget(self._stats_label)
        layout.addWidget(stats_group)

    def set_data(self, data: Any) -> bool:
        if not isinstance(data, CurveData):
            return False
        self._current_data = data
        self._channel_combo.blockSignals(True)
        self._channel_combo.clear()
        for channel_index, channel_name in enumerate(data.channel_names):
            self._channel_combo.addItem(channel_name, channel_index)
        self._channel_combo.setCurrentIndex(0)
        self._channel_combo.blockSignals(False)
        self._channel_combo.setEnabled(data.channels > 1)
        self._draw_channel(0)
        self.data_changed.emit()
        return True

    def _on_channel_changed(self, channel_index: int):
        if channel_index >= 0:
            self._draw_channel(channel_index)
            self.data_changed.emit()

    def _draw_channel(self, channel_index: int):
        data = self._current_data
        if not isinstance(data, CurveData) or not 0 <= channel_index < data.channels:
            return
        values = data.get_channel(channel_index)
        x_label = data.x_name + (f" ({data.x_unit})" if data.x_unit else "")
        y_label = data.y_name + (f" ({data.y_unit})" if data.y_unit else "")
        log_frequency = data.curve_type.startswith("frequency_octave_")

        self._update_data_info(
            f"📈 CURVE  shape={values.shape}  dtype={values.dtype}  "
            f"type={data.curve_type}  channel={data.channel_names[channel_index]}"
        )
        if HAS_PYQTGRAPH and self._plot_widget is not None:
            plot_item = self._plot_widget.getPlotItem()
            plot_item.setLogMode(x=log_frequency, y=False)
            plot_item.setLabel("bottom", x_label)
            plot_item.setLabel("left", y_label)
            self._curve.setData(data.x, values)
            plot_item.enableAutoRange()

        self._stats_label.setText(
            tr_("  Points: {count:,}\n").format(count=values.size)
            + tr_("  Minimum: {value:.6f}\n").format(value=float(np.min(values)))
            + tr_("  Maximum: {value:.6f}\n").format(value=float(np.max(values)))
            + tr_("  Mean: {value:.6f}\n").format(value=float(np.mean(values)))
            + tr_("  Standard deviation: {value:.6f}\n").format(value=float(np.std(values)))
            + tr_("  Median: {value:.6f}").format(value=float(np.median(values)))
        )

    def clear(self):
        self._current_data = None
        self._data_info = ""
        self._channel_combo.clear()
        self._stats_label.clear()
        if HAS_PYQTGRAPH and self._plot_widget is not None:
            self._curve.clear()

    @classmethod
    def can_display(cls, data: Any) -> bool:
        return isinstance(data, CurveData)


__all__ = ["CurvePreviewWidget"]
