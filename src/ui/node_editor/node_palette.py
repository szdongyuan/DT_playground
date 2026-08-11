# -*- coding: utf-8 -*-
"""
Node Palette Component

Displays all available workflow nodes with drag-and-drop to canvas support.
"""

import logging
from typing import Optional

from PySide6.QtCore import QMimeData, Qt, Signal
from PySide6.QtGui import QDrag
from PySide6.QtWidgets import (
    QFrame, QLabel, QLineEdit, QTreeWidget,
    QTreeWidgetItem, QVBoxLayout, QWidget
)

from src.ui.i18n import tr_
from src.ui.styles import Styles
from src.workflow.node_base import (
    NodeCategory, get_nodes_by_category
)

logger = logging.getLogger(__name__)


class NodeTreeWidget(QTreeWidget):
    """支持拖拽的节点树形控件"""
    
    node_double_clicked = Signal(str)  # node_type
    
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
            node_type = item.data(0, Qt.ItemDataRole.UserRole)
            
            drag = QDrag(self)
            mime_data = QMimeData()
            mime_data.setData("application/x-workflow-node", node_type.encode())
            drag.setMimeData(mime_data)
            drag.exec(Qt.DropAction.CopyAction)
    
    def _on_item_double_clicked(self, item, column):
        """双击添加节点"""
        node_type = item.data(0, Qt.ItemDataRole.UserRole)
        if node_type:
            self.node_double_clicked.emit(node_type)


class NodePalette(QWidget):
    """
    节点面板
    
    显示所有可用节点，按分类组织，支持搜索和拖拽。
    """
    
    node_selected = Signal(str)  # node_type
    node_add_requested = Signal(str)  # node_type
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._search_expansion_state = None
        self._empty_search_item = None
        self._setup_ui()
        self._populate_nodes()
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
        
        title = QLabel(tr_("🧩 Node Library"))
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
        self._search_input.setPlaceholderText(tr_("🔍 Search nodes..."))
        self._search_input.textChanged.connect(self._on_search)
        search_layout.addWidget(self._search_input)
        layout.addWidget(search_frame)
        
        # 节点树
        self._tree = NodeTreeWidget()
        self._tree.node_double_clicked.connect(self.node_add_requested)
        self._tree.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self._tree)
        
        # 提示
        hint = QLabel(tr_("Drag or double-click to add a node"))
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setStyleSheet(f"""
            color: {Styles.COLORS['subtext0']};
            font-size: 11px;
            padding: 8px;
        """)
        layout.addWidget(hint)
    
    def _populate_nodes(self):
        """填充节点列表"""
        self._tree.clear()
        
        node_count = 0
        
        for category in NodeCategory:
            node_classes = get_nodes_by_category(category)
            if not node_classes:
                continue
            
            # 创建分类节点
            category_item = QTreeWidgetItem([f"{category.display_name}"])
            category_item.setForeground(0, Styles.get_color(category.color))
            self._tree.addTopLevelItem(category_item)
            category_item.setExpanded(category == NodeCategory.DATA_SOURCE)
            
            # 按子类别分组节点
            subcategory_map = {}
            nodes_without_subcategory = []
            
            for node_class in node_classes:
                subcategory = getattr(node_class, 'subcategory', '')
                if subcategory:
                    if subcategory not in subcategory_map:
                        subcategory_map[subcategory] = []
                    subcategory_map[subcategory].append(node_class)
                else:
                    nodes_without_subcategory.append(node_class)
            
            # 添加无子类别的节点
            for node_class in sorted(
                nodes_without_subcategory,
                key=lambda cls: (getattr(cls, "palette_order", 100), cls.node_type),
            ):
                item = QTreeWidgetItem([f"{node_class.icon} {tr_(node_class.display_name)}"])
                item.setData(0, Qt.ItemDataRole.UserRole, node_class.node_type)
                item.setToolTip(0, tr_(node_class.description) if node_class.description else "")
                category_item.addChild(item)
                node_count += 1
            
            # 添加有子类别的节点
            grouped_subcategories = sorted(
                subcategory_map.items(),
                key=lambda item: (
                    min(getattr(cls, "subcategory_order", 100) for cls in item[1]),
                    item[0],
                ),
            )
            for subcategory, nodes in grouped_subcategories:
                # 创建子类别节点
                subcategory_item = QTreeWidgetItem([f"📁 {tr_(subcategory)}"])
                subcategory_item.setForeground(0, Styles.get_color(category.color))
                category_item.addChild(subcategory_item)
                subcategory_item.setExpanded(False)
                
                # 添加节点到子类别
                for node_class in sorted(
                    nodes,
                    key=lambda cls: (getattr(cls, "palette_order", 100), cls.node_type),
                ):
                    item = QTreeWidgetItem([f"{node_class.icon} {tr_(node_class.display_name)}"])
                    item.setData(0, Qt.ItemDataRole.UserRole, node_class.node_type)
                    item.setToolTip(0, tr_(node_class.description) if node_class.description else "")
                    subcategory_item.addChild(item)
                    node_count += 1
        
        logger.info(f"节点面板已加载 {node_count} 个节点")
    
    def _on_search(self, text: str):
        """Filter nodes by localized display name and manage expansion state."""
        query = text.casefold()

        if self._empty_search_item is not None:
            index = self._tree.indexOfTopLevelItem(self._empty_search_item)
            if index >= 0:
                self._tree.takeTopLevelItem(index)
            self._empty_search_item = None

        if not query:
            self._set_all_items_visible()
            self._restore_expansion_state()
            return

        if self._search_expansion_state is None:
            self._search_expansion_state = self._capture_expansion_state()

        match_count = 0

        for i in range(self._tree.topLevelItemCount()):
            category_item = self._tree.topLevelItem(i)
            category_visible = 0
            
            for j in range(category_item.childCount()):
                child = category_item.child(j)
                node_type = child.data(0, Qt.ItemDataRole.UserRole)
                
                if node_type:
                    # 这是一个节点项
                    match = query in child.text(0).casefold()
                    child.setHidden(not match)
                    if match:
                        category_visible += 1
                        match_count += 1
                else:
                    # 这是一个子类别项
                    subcategory_visible = 0
                    for k in range(child.childCount()):
                        node_item = child.child(k)
                        match = query in node_item.text(0).casefold()
                        node_item.setHidden(not match)
                        if match:
                            subcategory_visible += 1
                            match_count += 1
                    
                    child.setHidden(subcategory_visible == 0)
                    if subcategory_visible > 0:
                        child.setExpanded(True)
                        category_visible += 1
            
            category_item.setHidden(category_visible == 0)
            if category_visible > 0:
                category_item.setExpanded(True)

        if match_count == 0:
            self._empty_search_item = QTreeWidgetItem([tr_("No matching nodes")])
            self._empty_search_item.setDisabled(True)
            self._tree.addTopLevelItem(self._empty_search_item)

    def _capture_expansion_state(self):
        """Capture category and subcategory expansion state for search."""
        state = []
        for i in range(self._tree.topLevelItemCount()):
            category_item = self._tree.topLevelItem(i)
            child_state = [
                category_item.child(j).isExpanded()
                for j in range(category_item.childCount())
                if category_item.child(j).data(0, Qt.ItemDataRole.UserRole) is None
            ]
            state.append((category_item.isExpanded(), child_state))
        return state

    def _restore_expansion_state(self):
        """Restore the expansion state captured before search."""
        if self._search_expansion_state is None:
            return

        for i, (category_expanded, child_state) in enumerate(
            self._search_expansion_state
        ):
            category_item = self._tree.topLevelItem(i)
            category_item.setExpanded(category_expanded)
            subcategory_index = 0
            for j in range(category_item.childCount()):
                child = category_item.child(j)
                if child.data(0, Qt.ItemDataRole.UserRole) is None:
                    child.setExpanded(child_state[subcategory_index])
                    subcategory_index += 1

        self._search_expansion_state = None

    def _set_all_items_visible(self):
        """Show every category, subcategory, and node item."""
        for i in range(self._tree.topLevelItemCount()):
            category_item = self._tree.topLevelItem(i)
            category_item.setHidden(False)
            for j in range(category_item.childCount()):
                child = category_item.child(j)
                child.setHidden(False)
                for k in range(child.childCount()):
                    child.child(k).setHidden(False)
    
    def _on_item_clicked(self, item, column):
        """选中节点"""
        node_type = item.data(0, Qt.ItemDataRole.UserRole)
        if node_type:
            self.node_selected.emit(node_type)
    
    def _apply_styles(self):
        """应用样式"""
        self.setStyleSheet(f"""
            NodePalette {{
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

