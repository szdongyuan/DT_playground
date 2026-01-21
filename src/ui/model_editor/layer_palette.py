# -*- coding: utf-8 -*-
"""
Layer Palette Component

Displays all available neural network layers with drag-and-drop to canvas support.
"""

import logging
from typing import Optional

from PyQt6.QtCore import QMimeData, Qt, pyqtSignal
from PyQt6.QtGui import QDrag
from PyQt6.QtWidgets import (
    QFrame, QLabel, QLineEdit, QTreeWidget,
    QTreeWidgetItem, QVBoxLayout, QWidget
)

from src.model_builder.layer_base import (
    get_all_layer_types, get_layers_by_category, LayerCategory
)
from src.ui.styles import Styles

logger = logging.getLogger(__name__)


class LayerTreeWidget(QTreeWidget):
    """支持拖拽的层树形控件"""
    
    layer_double_clicked = pyqtSignal(str)  # layer_type
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setDragDropMode(QTreeWidget.DragDropMode.DragOnly)
        self.setDefaultDropAction(Qt.DropAction.CopyAction)
        self.setHeaderHidden(True)
        self.setIndentation(16)
        self.setAnimated(True)
        self.setRootIsDecorated(True)
        
        self._drag_start_pos = None
        
        self.itemDoubleClicked.connect(self._on_item_double_clicked)
    
    def mousePressEvent(self, event):
        """记录拖拽起始位置"""
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start_pos = event.pos()
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """检测拖拽动作"""
        if event.buttons() & Qt.MouseButton.LeftButton and self._drag_start_pos:
            distance = (event.pos() - self._drag_start_pos).manhattanLength()
            if distance >= 10:  # 拖拽阈值
                self._perform_drag()
                self._drag_start_pos = None
                return
        super().mouseMoveEvent(event)
    
    def _perform_drag(self):
        """执行拖拽操作"""
        item = self.currentItem()
        if item and item.data(0, Qt.ItemDataRole.UserRole):
            layer_type = item.data(0, Qt.ItemDataRole.UserRole)
            
            drag = QDrag(self)
            mime_data = QMimeData()
            mime_data.setData("application/x-model-layer", layer_type.encode())
            drag.setMimeData(mime_data)
            drag.exec(Qt.DropAction.CopyAction)
    
    def _on_item_double_clicked(self, item, column):
        """双击添加层"""
        layer_type = item.data(0, Qt.ItemDataRole.UserRole)
        if layer_type:
            self.layer_double_clicked.emit(layer_type)


class LayerPalette(QWidget):
    """
    层面板
    
    显示所有可用层，按分类组织，支持搜索和拖拽。
    """
    
    layer_selected = pyqtSignal(str)  # layer_type
    layer_add_requested = pyqtSignal(str)  # layer_type
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._populate_layers()
        self._apply_styles()
    
    def _setup_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # 标题
        header = QFrame()
        header.setFixedHeight(40)
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(12, 8, 12, 8)
        
        title = QLabel("🧱 层节点库")
        title.setStyleSheet(f"""
            font-size: 14px;
            font-weight: bold;
            color: {Styles.COLORS['text']};
        """)
        header_layout.addWidget(title)
        layout.addWidget(header)
        
        # 搜索框
        search_frame = QFrame()
        search_layout = QVBoxLayout(search_frame)
        search_layout.setContentsMargins(8, 4, 8, 8)
        
        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("🔍 搜索层...")
        self._search_input.textChanged.connect(self._on_search)
        search_layout.addWidget(self._search_input)
        layout.addWidget(search_frame)
        
        # 层树
        self._tree = LayerTreeWidget()
        self._tree.layer_double_clicked.connect(self.layer_add_requested)
        self._tree.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self._tree)
        
        # 提示
        hint = QLabel("拖拽或双击添加层")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setStyleSheet(f"""
            color: {Styles.COLORS['subtext0']};
            font-size: 11px;
            padding: 8px;
        """)
        layout.addWidget(hint)
    
    def _populate_layers(self):
        """填充层列表"""
        self._tree.clear()
        
        layer_count = 0
        
        for category in LayerCategory:
            layer_classes = get_layers_by_category(category)
            if not layer_classes:
                continue
            
            # 创建分类节点
            category_item = QTreeWidgetItem([f"{category.display_name}"])
            category_item.setForeground(0, Styles.get_color(category.color))
            category_item.setExpanded(True)
            self._tree.addTopLevelItem(category_item)
            
            # 添加层节点
            for layer_class in layer_classes:
                item = QTreeWidgetItem([f"{layer_class.icon} {layer_class.display_name}"])
                item.setData(0, Qt.ItemDataRole.UserRole, layer_class.layer_type)
                item.setToolTip(0, layer_class.description)
                category_item.addChild(item)
                layer_count += 1
        
        logger.info(f"层面板已加载 {layer_count} 个层")
    
    def _on_search(self, text: str):
        """搜索过滤"""
        text = text.lower()
        
        for i in range(self._tree.topLevelItemCount()):
            category_item = self._tree.topLevelItem(i)
            visible_children = 0
            
            for j in range(category_item.childCount()):
                child = category_item.child(j)
                match = text in child.text(0).lower()
                child.setHidden(not match)
                if match:
                    visible_children += 1
            
            category_item.setHidden(visible_children == 0)
    
    def _on_item_clicked(self, item, column):
        """选中层"""
        layer_type = item.data(0, Qt.ItemDataRole.UserRole)
        if layer_type:
            self.layer_selected.emit(layer_type)
    
    def _apply_styles(self):
        """应用样式"""
        self.setStyleSheet(f"""
            LayerPalette {{
                background: {Styles.COLORS['mantle']};
                border-right: 1px solid {Styles.COLORS['surface0']};
            }}
            
            QLineEdit {{
                background: {Styles.COLORS['surface0']};
                border: 1px solid {Styles.COLORS['surface1']};
                border-radius: 6px;
                padding: 6px 10px;
                color: {Styles.COLORS['text']};
            }}
            
            QLineEdit:focus {{
                border-color: {Styles.COLORS['blue']};
            }}
            
            QTreeWidget {{
                background: transparent;
                border: none;
                outline: none;
            }}
            
            QTreeWidget::item {{
                padding: 6px 8px;
                border-radius: 4px;
            }}
            
            QTreeWidget::item:hover {{
                background: {Styles.COLORS['surface0']};
            }}
            
            QTreeWidget::item:selected {{
                background: {Styles.COLORS['surface1']};
                color: {Styles.COLORS['blue']};
            }}
        """)

