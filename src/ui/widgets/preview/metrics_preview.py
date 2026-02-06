# -*- coding: utf-8 -*-
"""
Metrics Preview Component

Displays visualization of evaluation metrics.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QGroupBox, QHBoxLayout, QLabel, QSplitter, QVBoxLayout, QWidget
)

from src.ui.i18n import tr_
from src.ui.styles import Styles

from .base_preview import BasePreviewWidget, register_preview


logger = logging.getLogger(__name__)


try:
    import pyqtgraph as pg
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False


@register_preview
class MetricsPreviewWidget(BasePreviewWidget):
    """
    指标预览组件
    
    显示评估指标的柱状图和详细文本。
    """
    
    display_name = tr_("Metrics")
    icon = "📊"
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # 主分割器
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)
        layout.addWidget(splitter)
        
        # 左侧：柱状图
        chart_group = QGroupBox(tr_("📊 Metrics"))
        chart_group.setStyleSheet(Styles.group_box(Styles.COLORS['blue']))
        chart_layout = QVBoxLayout(chart_group)
        chart_layout.setContentsMargins(4, 4, 4, 4)
        
        if HAS_PYQTGRAPH:
            self._bar_widget = pg.PlotWidget()
            self._bar_widget.setBackground('#181825')
            self._bar_widget.showGrid(x=False, y=True, alpha=0.3)
            self._bar_widget.setLabel('left', text=tr_('Value'))
            self._bar_widget.setLabel('bottom', text=tr_('Metric'))
            self._bar_widget.setMinimumHeight(300)
            chart_layout.addWidget(self._bar_widget)
        else:
            placeholder = QLabel(tr_("Please install pyqtgraph to display charts"))
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            chart_layout.addWidget(placeholder)
        
        splitter.addWidget(chart_group)
        
        # 右侧：详细指标文本
        detail_group = QGroupBox(tr_("📋 Details"))
        detail_group.setStyleSheet(Styles.group_box(Styles.COLORS['peach']))
        detail_layout = QVBoxLayout(detail_group)
        detail_layout.setContentsMargins(12, 12, 12, 12)
        
        self._metrics_label = QLabel()
        self._metrics_label.setStyleSheet(f"""
            QLabel {{
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 14px;
                color: {Styles.COLORS['text']};
                background: {Styles.COLORS['mantle']};
                padding: 16px;
                border-radius: 8px;
                line-height: 1.8;
            }}
        """)
        self._metrics_label.setWordWrap(True)
        self._metrics_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        detail_layout.addWidget(self._metrics_label)
        
        splitter.addWidget(detail_group)
        
        # 设置分割器比例
        splitter.setSizes([400, 300])
    
    def set_data(self, data: Any) -> bool:
        """
        设置指标数据
        
        Args:
            data: 指标字典 {name: value}
            
        Returns:
            是否成功设置
        """
        self._current_data = data
        
        if data is None:
            self.clear()
            return False
        
        if not isinstance(data, dict):
            logger.warning(f"不支持的指标数据类型: {type(data)}")
            return False
        
        try:
            # 更新数据信息
            num_metrics = len(data)
            self._update_data_info(
                f"📊 METRICS  count={num_metrics}  keys={list(data.keys())}"
            )
            
            # 更新柱状图
            self._update_chart(data)
            
            # 更新详细文本
            self._update_details(data)
            
            self.data_changed.emit()
            return True
            
        except Exception as e:
            logger.error(f"设置指标数据失败: {e}")
            return False
    
    def _update_chart(self, metrics: Dict):
        """更新柱状图"""
        if not HAS_PYQTGRAPH:
            return
        
        self._bar_widget.clear()
        
        # 只显示数值类型的指标
        numeric_metrics = {
            k: v for k, v in metrics.items()
            if isinstance(v, (int, float, np.integer, np.floating))
        }
        
        if not numeric_metrics:
            return
        
        names = list(numeric_metrics.keys())
        values = list(numeric_metrics.values())
        
        x = np.arange(len(names))
        
        # 根据指标名选择颜色
        colors = []
        for name in names:
            name_lower = name.lower()
            if 'loss' in name_lower:
                colors.append('#f38ba8')  # 红色
            elif 'accuracy' in name_lower or 'acc' in name_lower:
                colors.append('#a6e3a1')  # 绿色
            elif 'precision' in name_lower:
                colors.append('#89b4fa')  # 蓝色
            elif 'recall' in name_lower:
                colors.append('#cba6f7')  # 紫色
            elif 'f1' in name_lower:
                colors.append('#f9e2af')  # 黄色
            else:
                colors.append('#94e2d5')  # 青色
        
        brushes = [pg.mkBrush(c) for c in colors]
        
        bar_item = pg.BarGraphItem(
            x=x, height=values, width=0.6, brushes=brushes
        )
        self._bar_widget.addItem(bar_item)
        
        # 设置X轴标签
        axis = self._bar_widget.getPlotItem().getAxis('bottom')
        axis.setTicks([[(i, name) for i, name in enumerate(names)]])
        
        # 设置范围
        self._bar_widget.setXRange(-0.5, len(names) - 0.5)
        if values:
            max_val = max(abs(v) for v in values)
            min_val = min(0, min(values))
            self._bar_widget.setYRange(min_val, max_val * 1.15)
    
    def _update_details(self, metrics: Dict):
        """更新详细文本"""
        lines = []
        lines.append("═" * 40)
        lines.append("")
        
        for name, value in metrics.items():
            if isinstance(value, float):
                name_lower = name.lower()
                if 'accuracy' in name_lower or 'acc' in name_lower:
                    line = f"  <b>{name}:</b>  {value:.4f}  ({value*100:.2f}%)"
                elif 'loss' in name_lower:
                    line = f"  <b>{name}:</b>  {value:.6f}"
                else:
                    line = f"  <b>{name}:</b>  {value:.6f}"
            else:
                line = f"  <b>{name}:</b>  {value}"
            lines.append(line)
        
        lines.append("")
        lines.append("═" * 40)
        
        self._metrics_label.setText("<br>".join(lines))
        
        # 同时输出到日志
        logger.info("=" * 40)
        logger.info("📊 Metrics:")
        for name, value in metrics.items():
            if isinstance(value, float):
                if 'accuracy' in name.lower() or 'acc' in name.lower():
                    logger.info(f"  {name}: {value:.4f} ({value*100:.2f}%)")
                else:
                    logger.info(f"  {name}: {value:.6f}")
            else:
                logger.info(f"  {name}: {value}")
        logger.info("=" * 40)
    
    def clear(self):
        """清除显示"""
        self._current_data = None
        self._data_info = ""
        if HAS_PYQTGRAPH:
            self._bar_widget.clear()
        self._metrics_label.setText("")
    
    @classmethod
    def can_display(cls, data: Any) -> bool:
        """检查是否可以显示该数据"""
        if not isinstance(data, dict):
            return False
        
        if len(data) == 0:
            return False
        
        # 检查是否所有值都是数值类型
        # （区别于标签字典，标签字典的值可能是字符串）
        numeric_count = sum(
            1 for v in data.values()
            if isinstance(v, (int, float, np.integer, np.floating))
        )
        
        # 如果大多数值是数值且没有标签相关的键
        if numeric_count == len(data):
            # 排除标签字典
            if 'labels' in data or 'filename_map' in data:
                return False
            return True
        
        return False

