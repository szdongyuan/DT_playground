# -*- coding: utf-8 -*-
"""
标签预览组件

显示标签数据的表格和统计信息。
"""

import logging
from collections import Counter
from typing import Any, Dict, List, Optional, Union

import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget
)

from src.workflow.port import DataType
from src.ui.styles import Styles

from .base_preview import BasePreviewWidget, register_preview


logger = logging.getLogger(__name__)


try:
    import pyqtgraph as pg
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False


@register_preview
class LabelPreviewWidget(BasePreviewWidget):
    """
    标签预览组件
    
    显示标签数据的表格视图、统计信息和分布图。
    """
    
    supported_types = [DataType.LABEL]
    display_name = "标签预览"
    icon = "🏷️"
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._label_list: List = []
        self._label_map: Dict = {}
        self._setup_ui()
    
    def _setup_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # 主分割器
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setHandleWidth(2)
        layout.addWidget(splitter)
        
        # 上半部分：统计信息和分布图
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(8)
        
        # 统计信息卡片
        stats_group = QGroupBox("📊 统计信息")
        stats_group.setStyleSheet(Styles.group_box(Styles.COLORS['teal']))
        stats_layout = QVBoxLayout(stats_group)
        stats_layout.setContentsMargins(12, 12, 12, 12)
        
        self._stats_label = QLabel()
        self._stats_label.setStyleSheet(f"""
            QLabel {{
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 13px;
                color: {Styles.COLORS['text']};
                line-height: 1.6;
            }}
        """)
        self._stats_label.setWordWrap(True)
        stats_layout.addWidget(self._stats_label)
        stats_layout.addStretch()
        top_layout.addWidget(stats_group, 1)
        
        # 分布柱状图
        dist_group = QGroupBox("📈 标签分布")
        dist_group.setStyleSheet(Styles.group_box(Styles.COLORS['lavender']))
        dist_layout = QVBoxLayout(dist_group)
        dist_layout.setContentsMargins(4, 4, 4, 4)
        
        if HAS_PYQTGRAPH:
            self._bar_widget = pg.PlotWidget()
            self._bar_widget.setBackground('#181825')
            self._bar_widget.showGrid(x=False, y=True, alpha=0.3)
            self._bar_widget.setLabel('left', text='数量')
            self._bar_widget.setLabel('bottom', text='类别')
            self._bar_widget.setMinimumHeight(180)
            dist_layout.addWidget(self._bar_widget)
        else:
            placeholder = QLabel("请安装 pyqtgraph 以显示分布图")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            dist_layout.addWidget(placeholder)
        
        top_layout.addWidget(dist_group, 2)
        splitter.addWidget(top_widget)
        
        # 下半部分：标签列表表格
        table_group = QGroupBox("📋 标签列表")
        table_group.setStyleSheet(Styles.group_box(Styles.COLORS['blue']))
        table_layout = QVBoxLayout(table_group)
        table_layout.setContentsMargins(4, 4, 4, 4)
        
        self._table = QTableWidget()
        self._table.setColumnCount(2)
        self._table.setHorizontalHeaderLabels(["类别名", "标签值"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.setAlternatingRowColors(True)
        self._table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {Styles.COLORS['mantle']};
                border: 1px solid {Styles.COLORS['surface1']};
                border-radius: 4px;
                color: {Styles.COLORS['text']};
                gridline-color: {Styles.COLORS['surface0']};
            }}
            QTableWidget::item {{
                padding: 4px;
            }}
            QTableWidget::item:selected {{
                background-color: {Styles.COLORS['surface1']};
            }}
            QTableWidget::item:alternate {{
                background-color: {Styles.COLORS['base']};
            }}
            QHeaderView::section {{
                background-color: {Styles.COLORS['surface0']};
                color: {Styles.COLORS['text']};
                padding: 6px;
                border: none;
                font-weight: bold;
            }}
        """)
        table_layout.addWidget(self._table)
        splitter.addWidget(table_group)
        
        # 设置分割器比例
        splitter.setSizes([250, 350])
    
    def set_data(self, data: Any) -> bool:
        """
        设置标签数据
        
        支持的数据格式：
        - List[int/str]: 标签列表
        - Dict: 标签字典 {filename: label} 或带元数据的字典
        - 单个标签值 (int/str)
        
        Args:
            data: 标签数据
            
        Returns:
            是否成功设置
        """
        self._current_data = data
        
        if data is None:
            self.clear()
            return False
        
        try:
            # 解析标签数据
            labels, label_map = self._parse_labels(data)
            
            if not labels:
                logger.warning("没有可显示的标签数据")
                self.clear()
                return False
            
            self._label_list = labels
            self._label_map = label_map
            
            # 更新数据信息
            unique_count = len(set(labels))
            self._update_data_info(
                f"🏷️ LABEL  count={len(labels)}  classes={unique_count}"
            )
            
            # 更新UI
            self._update_stats(labels, label_map)
            self._update_distribution(labels)
            self._update_table(labels, label_map)
            
            self.data_changed.emit()
            return True
            
        except Exception as e:
            logger.error(f"设置标签数据失败: {e}")
            return False
    
    def _parse_labels(self, data: Any) -> tuple:
        """
        解析标签数据
        
        Returns:
            (labels_list, label_map_dict)
        """
        labels = []
        label_map = {}
        
        if isinstance(data, list):
            # 直接是标签列表
            labels = data
        elif isinstance(data, dict):
            # 字典格式
            if 'labels' in data:
                # SaveAudioNode 格式: {"labels": {filename: label}, ...}
                label_dict = data['labels']
                if isinstance(label_dict, dict):
                    label_map = label_dict
                    labels = list(label_dict.values())
                else:
                    labels = label_dict
            elif 'filename_map' in data:
                # LabelFileNode 格式
                label_map = data.get('filename_map', {})
                labels = list(label_map.values())
            else:
                # 普通字典 {filename: label}
                label_map = data
                labels = list(data.values())
        elif isinstance(data, (int, str, float)):
            # 单个标签
            labels = [data]
        elif isinstance(data, np.ndarray):
            labels = data.tolist()
        else:
            logger.warning(f"不支持的标签数据类型: {type(data)}")
        
        return labels, label_map
    
    def _update_stats(self, labels: List, label_map: Dict):
        """更新统计信息"""
        total = len(labels)
        counter = Counter(labels)
        unique_count = len(counter)
        
        # 找出最多和最少的类别
        if counter:
            most_common = counter.most_common(1)[0]
            least_common = counter.most_common()[-1]
        else:
            most_common = ("-", 0)
            least_common = ("-", 0)
        
        stats_text = f"""
<b>总标签数:</b> {total:,}

<b>类别数:</b> {unique_count}

<b>最多类别:</b> {most_common[0]} ({most_common[1]:,} 个, {most_common[1]/total*100:.1f}%)

<b>最少类别:</b> {least_common[0]} ({least_common[1]:,} 个, {least_common[1]/total*100:.1f}%)

<b>平均每类:</b> {total/unique_count:.1f} 个
        """
        
        self._stats_label.setText(stats_text.strip())
    
    def _update_distribution(self, labels: List):
        """更新分布柱状图"""
        if not HAS_PYQTGRAPH:
            return
        
        self._bar_widget.clear()
        
        counter = Counter(labels)
        sorted_items = sorted(counter.items(), key=lambda x: x[0])
        
        if not sorted_items:
            return
        
        names = [str(item[0]) for item in sorted_items]
        values = [item[1] for item in sorted_items]
        
        # 创建柱状图
        x = np.arange(len(names))
        
        # 使用不同颜色
        colors = [
            '#89b4fa', '#a6e3a1', '#f9e2af', '#fab387', 
            '#f38ba8', '#cba6f7', '#94e2d5', '#f5c2e7'
        ]
        brushes = [pg.mkBrush(colors[i % len(colors)]) for i in range(len(names))]
        
        bar_item = pg.BarGraphItem(
            x=x, height=values, width=0.6, brushes=brushes
        )
        self._bar_widget.addItem(bar_item)
        
        # 设置X轴标签
        axis = self._bar_widget.getPlotItem().getAxis('bottom')
        axis.setTicks([[(i, name) for i, name in enumerate(names)]])
        
        # 设置范围
        self._bar_widget.setXRange(-0.5, len(names) - 0.5)
        self._bar_widget.setYRange(0, max(values) * 1.1)
    
    def _update_table(self, labels: List, label_map: Dict):
        """更新表格"""
        self._table.setRowCount(0)
        
        # 限制显示数量，避免太多数据导致卡顿
        max_display = 1000
        display_count = min(len(labels), max_display)
        
        self._table.setRowCount(display_count)
        
        # 如果有文件名映射，使用它
        filenames = list(label_map.keys()) if label_map else []
        
        for i in range(display_count):
            # 类别名（如果有文件名则显示文件名）
            if i < len(filenames):
                name = filenames[i]
            else:
                name = f"item_{i}"
            name_item = QTableWidgetItem(name)
            self._table.setItem(i, 0, name_item)
            
            # 标签值
            label_value = labels[i]
            value_item = QTableWidgetItem(str(label_value))
            value_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(i, 1, value_item)
        
        # 如果数据被截断，添加提示
        if len(labels) > max_display:
            self._table.setRowCount(display_count + 1)
            hint_item = QTableWidgetItem(f"... 还有 {len(labels) - max_display} 条数据未显示")
            hint_item.setForeground(Styles.get_color('subtext0'))
            self._table.setItem(display_count, 0, hint_item)
            self._table.setSpan(display_count, 0, 1, 3)
    
    def clear(self):
        """清除显示"""
        self._current_data = None
        self._data_info = ""
        self._label_list = []
        self._label_map = {}
        self._stats_label.setText("")
        self._table.setRowCount(0)
        if HAS_PYQTGRAPH:
            self._bar_widget.clear()
    
    @classmethod
    def can_display(cls, data: Any) -> bool:
        """检查是否可以显示该数据"""
        # 标签列表
        if isinstance(data, list):
            if len(data) == 0:
                return True
            # 检查是否是简单类型的列表（非对象列表）
            first = data[0]
            if isinstance(first, (int, str, float)):
                return True
            # numpy 整数
            if isinstance(first, (np.integer, np.floating)):
                return True
            return False
        
        # 标签字典
        if isinstance(data, dict):
            # SaveAudioNode 格式
            if 'labels' in data:
                return True
            # LabelFileNode 格式
            if 'filename_map' in data or 'label_names' in data:
                return True
            # 普通 {filename: label} 字典
            if len(data) > 0:
                first_value = next(iter(data.values()))
                if isinstance(first_value, (int, str, float, np.integer)):
                    return True
        
        # 单个标签值
        if isinstance(data, (int, str)):
            return True
        
        return False

