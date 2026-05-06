# -*- coding: utf-8 -*-
"""
Preview View

Displays audio waveform, spectrogram, features, labels, and other data previews.
使用组件化架构，支持多种数据类型的可视化。
"""

import logging
import os
from typing import Any, Optional, Type

import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox, QFrame, QHBoxLayout, QLabel,
    QPushButton, QStackedWidget, QVBoxLayout, QWidget
)

from ..i18n import tr_
from ..styles import Styles
from ..widgets.preview import (
    BasePreviewWidget,
    AudioPreviewWidget,
    Feature1DPreviewWidget,
    Feature2DPreviewWidget,
    LabelPreviewWidget,
    MetricsPreviewWidget,
    ModelPreviewWidget,
)


logger = logging.getLogger(__name__)


class PreviewView(QWidget):
    """
    预览视图
    
    显示选中节点的输出数据，根据数据类型自动切换预览组件。
    
    Signals:
        preview_requested: 请求预览指定节点 (node_id)
    """
    
    preview_requested = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._current_data: Any = None
        self._current_data_list: list = []  # 当数据是列表时保存完整列表
        self._current_node_id: Optional[str] = None
        
        self._setup_ui()
        self._register_preview_widgets()
    
    def _setup_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # 标题栏
        header = self._create_header()
        layout.addWidget(header)
        
        # 堆叠组件 - 管理不同类型的预览面板
        self._stack = QStackedWidget()
        layout.addWidget(self._stack, 1)  # stretch=1 让它占据剩余空间
        
        # 空白占位页面
        self._empty_widget = QWidget()
        empty_layout = QVBoxLayout(self._empty_widget)
        empty_label = QLabel(tr_("Select a node and run the workflow to preview data"))
        empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_label.setStyleSheet(f"""
            color: {Styles.COLORS['subtext0']};
            font-size: 14px;
        """)
        empty_layout.addWidget(empty_label)
        self._stack.addWidget(self._empty_widget)
    
    def _register_preview_widgets(self):
        """注册所有预览组件"""
        self._preview_widgets: dict[str, BasePreviewWidget] = {}
        
        # 创建并注册各类型预览组件
        preview_classes = [
            ('audio', AudioPreviewWidget),
            ('feature_1d', Feature1DPreviewWidget),
            ('feature_2d', Feature2DPreviewWidget),
            ('label', LabelPreviewWidget),
            ('metrics', MetricsPreviewWidget),
            ('model', ModelPreviewWidget),
        ]
        
        for name, cls in preview_classes:
            widget = cls()
            self._preview_widgets[name] = widget
            self._stack.addWidget(widget)
        
        # 默认显示空白页面
        self._stack.setCurrentWidget(self._empty_widget)
        
        # ===== 快捷键设置 =====
        # Page Up - 上一个数据项
        prev_shortcut = QShortcut(QKeySequence(Qt.Key.Key_PageUp), self)
        prev_shortcut.activated.connect(self._go_to_prev_item)
        
        # Page Down - 下一个数据项
        next_shortcut = QShortcut(QKeySequence(Qt.Key.Key_PageDown), self)
        next_shortcut.activated.connect(self._go_to_next_item)
    
    def _create_header(self) -> QWidget:
        """创建标题栏"""
        header = QFrame()
        header.setStyleSheet(f"""
            QFrame {{
                background: {Styles.COLORS['surface0']};
                border-radius: 6px;
                padding: 8px;
            }}
        """)
        
        layout = QHBoxLayout(header)
        layout.setContentsMargins(8, 4, 8, 4)
        
        # 标题
        title = QLabel(tr_("📊 Data Preview"))
        title.setStyleSheet(f"""
            font-size: 14px;
            font-weight: bold;
            color: {Styles.COLORS['text']};
        """)
        layout.addWidget(title)
        
        # 当前节点信息
        self._node_info = QLabel(tr_("No node selected"))
        self._node_info.setStyleSheet(f"color: {Styles.COLORS['subtext1']};")
        layout.addWidget(self._node_info)
        
        # 数据维度信息
        self._dim_info = QLabel("")
        self._dim_info.setStyleSheet(f"""
            color: {Styles.COLORS['green']};
            font-family: 'Consolas', 'Courier New', monospace;
            font-weight: bold;
            background: {Styles.COLORS['surface1']};
            padding: 2px 8px;
            border-radius: 4px;
        """)
        layout.addWidget(self._dim_info)
        
        layout.addStretch()
        
        # ===== 数据项索引选择 =====
        self._item_label = QLabel(tr_("Item:"))
        self._item_label.setStyleSheet(f"color: {Styles.COLORS['subtext1']};")
        layout.addWidget(self._item_label)
        
        self._item_combo = QComboBox()
        self._item_combo.setMinimumWidth(180)
        self._item_combo.currentIndexChanged.connect(self._on_item_changed)
        self._item_combo.setStyleSheet(Styles.FORM_CONTROLS)
        layout.addWidget(self._item_combo)
        
        # 默认隐藏数据项选择器
        self._item_label.hide()
        self._item_combo.hide()
        
        # 输出端口选择
        port_label = QLabel(tr_("Output port:"))
        port_label.setStyleSheet(f"color: {Styles.COLORS['subtext1']};")
        layout.addWidget(port_label)
        
        self._port_combo = QComboBox()
        self._port_combo.setMinimumWidth(120)
        self._port_combo.currentIndexChanged.connect(self._on_port_changed)
        self._port_combo.setStyleSheet(Styles.FORM_CONTROLS)
        layout.addWidget(self._port_combo)
        
        # 刷新按钮
        refresh_btn = QPushButton(tr_("🔄 Refresh"))
        refresh_btn.clicked.connect(self._on_refresh)
        refresh_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Styles.COLORS['surface1']};
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                color: {Styles.COLORS['text']};
            }}
            QPushButton:hover {{
                background: {Styles.COLORS['surface2']};
            }}
        """)
        layout.addWidget(refresh_btn)
        
        return header
    
    def set_node_data(self, node_id: str, node_name: str, outputs: dict):
        """
        设置节点输出数据
        
        Args:
            node_id: 节点ID
            node_name: 节点名称
            outputs: 输出端口数据字典 {port_name: data}
        """
        self._current_node_id = node_id
        self._node_info.setText(
            tr_("Node: {name} ({id})").format(name=node_name, id=node_id)
        )
        
        # 更新端口下拉框
        self._port_combo.blockSignals(True)
        self._port_combo.clear()
        for port_name, data in outputs.items():
            if data is not None:
                self._port_combo.addItem(port_name, data)
        self._port_combo.blockSignals(False)
        
        # 显示第一个端口的数据
        if self._port_combo.count() > 0:
            self._display_data(self._port_combo.currentData())
        else:
            # 没有数据时显示提示
            self._clear_display()
            self._node_info.setText(
                tr_("Node: {name} (no data - run workflow first)").format(name=node_name)
            )
    
    def _on_port_changed(self, index: int):
        """端口选择变化"""
        if index >= 0:
            data = self._port_combo.currentData()
            self._display_data(data)
    
    def _on_item_changed(self, index: int):
        """数据项选择变化"""
        if index >= 0 and self._current_data_list:
            if index < len(self._current_data_list):
                item = self._current_data_list[index]
                self._display_single_item(item)
    
    def _go_to_prev_item(self):
        """切换到上一个数据项 (Page Up)"""
        if not self._current_data_list or self._item_combo.count() == 0:
            return
        
        current_index = self._item_combo.currentIndex()
        if current_index > 0:
            self._item_combo.setCurrentIndex(current_index - 1)
    
    def _go_to_next_item(self):
        """切换到下一个数据项 (Page Down)"""
        if not self._current_data_list or self._item_combo.count() == 0:
            return
        
        current_index = self._item_combo.currentIndex()
        if current_index < self._item_combo.count() - 1:
            self._item_combo.setCurrentIndex(current_index + 1)
    
    def _on_refresh(self):
        """刷新预览"""
        if self._current_node_id:
            self.preview_requested.emit(self._current_node_id)
    
    def _display_data(self, data: Any):
        """显示数据"""
        self._current_data = data
        
        if data is None:
            self._clear_display()
            return
        
        # 处理列表数据
        if isinstance(data, list) and len(data) > 0:
            # 检查是否是标签列表（简单类型的列表）
            # 标签列表应该整体显示分布，而不是逐个预览
            if self._is_label_list(data):
                self._current_data_list = []
                self._item_label.hide()
                self._item_combo.hide()
                # 直接将整个列表传递给 LabelPreviewWidget
                self._display_single_item(data)
            else:
                # 其他列表（如音频列表、特征列表）使用数据项选择器
                self._current_data_list = data
                self._setup_item_selector(data)
                # 显示当前选中的项
                selected_index = self._item_combo.currentIndex()
                if selected_index < 0 or selected_index >= len(data):
                    selected_index = 0
                self._display_single_item(data[selected_index])
        else:
            # 非列表数据
            self._current_data_list = []
            self._item_label.hide()
            self._item_combo.hide()
            self._display_single_item(data)
    
    def _is_label_list(self, data: list) -> bool:
        """
        检查是否是标签列表
        
        标签列表的特点：元素是简单类型（int, str, float）
        """
        if not data:
            return False
        
        first = data[0]
        # 简单类型的列表视为标签列表
        if isinstance(first, (int, str, float)):
            return True
        # numpy 数值类型
        if isinstance(first, (np.integer, np.floating)):
            return True
        return False
    
    def _setup_item_selector(self, data_list: list):
        """设置数据项选择器"""
        self._item_label.show()
        self._item_combo.show()
        
        # 阻止信号避免重复触发
        self._item_combo.blockSignals(True)
        self._item_combo.clear()
        
        # 填充选项
        for i, item in enumerate(data_list):
            label = self._get_item_label(i, item)
            self._item_combo.addItem(label, i)
        
        self._item_combo.blockSignals(False)
        
        # 更新总数显示
        total = len(data_list)
        self._item_combo.setToolTip(tr_("Total: {total} item(s)").format(total=total))
    
    def _get_item_label(self, index: int, item: Any) -> str:
        """获取数据项的显示标签"""
        if hasattr(item, 'file_path') and item.file_path:
            # AudioData 有 file_path 属性
            name = os.path.basename(item.file_path)
            return f"[{index}] {name}"
        elif hasattr(item, 'feature_type'):
            # FeatureData
            return f"[{index}] {item.feature_type}"
        elif isinstance(item, np.ndarray):
            return f"[{index}] array{item.shape}"
        elif isinstance(item, (int, str, float)):
            # 简单标签值
            return f"[{index}] = {item}"
        else:
            return f"[{index}]"
    
    def _display_single_item(self, data: Any):
        """显示单个数据项"""
        # 确定使用哪个预览组件
        preview_widget = self._get_preview_widget_for_data(data)
        
        if preview_widget is None:
            logger.warning(f"没有找到适合的预览组件: {type(data)}")
            self._stack.setCurrentWidget(self._empty_widget)
            self._dim_info.setText(
                tr_("Unsupported type: {type}").format(type=type(data).__name__)
            )
            return
        
        # 设置数据到预览组件
        if preview_widget.set_data(data):
            self._stack.setCurrentWidget(preview_widget)
            self._dim_info.setText(preview_widget.data_info)
        else:
            self._stack.setCurrentWidget(self._empty_widget)
            self._dim_info.setText(tr_("Failed to set data"))
    
    def _get_preview_widget_for_data(self, data: Any) -> Optional[BasePreviewWidget]:
        """根据数据类型获取合适的预览组件"""
        # 按优先级检查各个预览组件
        # 顺序很重要：先检查更具体的类型
        
        # 0. 检查是否是 Keras 模型
        if ModelPreviewWidget.can_display(data):
            return self._preview_widgets.get('model')
        
        # 1. 检查是否是 FeatureData
        if hasattr(data, 'data') and hasattr(data, 'feature_type'):
            arr = data.data
            # 判断是1D还是2D特征
            if arr.ndim == 1 or (arr.ndim == 2 and arr.shape[0] <= 2):
                return self._preview_widgets.get('feature_1d')
            else:
                return self._preview_widgets.get('feature_2d')
        
        # 2. 检查是否是 AudioData
        if hasattr(data, 'data') and hasattr(data, 'sample_rate'):
            return self._preview_widgets.get('audio')
        
        # 3. 检查是否是标签数据
        if LabelPreviewWidget.can_display(data):
            return self._preview_widgets.get('label')
        
        # 4. 检查是否是评估指标
        if MetricsPreviewWidget.can_display(data):
            return self._preview_widgets.get('metrics')
        
        # 5. 检查是否是 numpy 数组
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                return self._preview_widgets.get('feature_1d')
            elif data.ndim >= 2:
                return self._preview_widgets.get('feature_2d')
        
        return None
    
    def _clear_display(self):
        """清除显示"""
        # 清除所有预览组件的数据
        for widget in self._preview_widgets.values():
            widget.clear()
        
        # 显示空白页面
        self._stack.setCurrentWidget(self._empty_widget)
        
        # 隐藏数据项选择器
        self._item_label.hide()
        self._item_combo.hide()
        self._current_data_list = []
        
        self._dim_info.setText("")
    
    def clear(self):
        """清除所有"""
        self._current_data = None
        self._current_data_list = []
        self._current_node_id = None
        self._node_info.setText(tr_("No node selected"))
        self._port_combo.clear()
        self._clear_display()
