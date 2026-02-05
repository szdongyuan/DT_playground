# -*- coding: utf-8 -*-
"""
Node Canvas Component

Node editing canvas implemented with pure PyQt6.

架构说明
--------
- PortItem 继承自 BasePortItem（graph_editor 基类）
- ConnectionItem 继承自 BaseConnectionItem
- NodeGraphScene 继承自 BaseGraphScene
- NodeGraphView 继承自 BaseGraphView
"""

import logging
from typing import Dict, List, Optional, Tuple

from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QAction, QBrush, QColor, QDragEnterEvent, QDropEvent,
    QFont, QKeyEvent, QMouseEvent, QPainter, QPainterPath, QPen, QWheelEvent
)
from PyQt6.QtWidgets import (
    QGraphicsItem, QGraphicsRectItem,
    QGraphicsTextItem, QMenu, QVBoxLayout, QWidget
)

from ..styles import Styles
from ..graph_editor import BasePortItem, BaseConnectionItem, BaseGraphScene, BaseGraphView
from ..i18n import tr_
from ...workflow.connection import Connection
from ...workflow.node_base import (
    BaseNode, NodeCategory, create_node, get_all_node_types,
    get_node_class, get_nodes_by_category
)
from ...workflow.workflow import Workflow


logger = logging.getLogger(__name__)


# ===== 图形节点 =====

class PortItem(BasePortItem):
    """
    工作流节点端口图形项
    
    继承自 BasePortItem，增加端口名称属性。
    """
    
    def __init__(self, port_name: str, is_input: bool, color: str, parent=None):
        super().__init__(is_input, color, parent)
        self.port_name = port_name


class NodeItem(QGraphicsRectItem):
    """节点图形项"""
    
    NODE_WIDTH = 180
    NODE_MIN_HEIGHT = 80
    HEADER_HEIGHT = 28
    PORT_SPACING = 24
    PORT_MARGIN = 12
    
    # 紧凑模式尺寸
    COMPACT_WIDTH = 60
    COMPACT_HEIGHT = 36
    
    def __init__(self, node: BaseNode, scene: 'NodeGraphScene'):
        super().__init__()
        self.node = node
        self.node_scene = scene
        self.input_ports: Dict[str, PortItem] = {}
        self.output_ports: Dict[str, PortItem] = {}
        
        # 执行状态 (idle, running, completed, error, waiting)
        self._execution_state: str = 'idle'
        self._state_icon_item: Optional[QGraphicsTextItem] = None
        
        # 检查是否为紧凑模式
        self.is_compact = getattr(node, 'compact_mode', False)
        
        if self.is_compact:
            # 紧凑模式：小尺寸
            width = self.COMPACT_WIDTH
            height = self.COMPACT_HEIGHT
        else:
            # 标准模式：根据端口数计算高度
            max_ports = max(len(node.inputs), len(node.outputs), 1)
            width = self.NODE_WIDTH
            height = self.HEADER_HEIGHT + max_ports * self.PORT_SPACING + self.PORT_MARGIN
        
        # 设置矩形
        self.setRect(0, 0, width, height)
        
        # 设置属性
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)
        
        # 设置位置
        self.setPos(node.position[0], node.position[1])
        
        # 颜色
        self.category_color = node.category.color
        
        # 创建子元素
        self._create_header()
        self._create_ports()
    
    def _create_header(self):
        """创建头部"""
        if self.is_compact:
            # 紧凑模式：只显示图标，居中
            title = QGraphicsTextItem(self.node.icon, self)
            title.setDefaultTextColor(QColor("#cdd6f4"))
            title.setFont(QFont("Microsoft YaHei", 12, QFont.Weight.Bold))
            # 居中显示
            text_width = title.boundingRect().width()
            title.setPos((self.COMPACT_WIDTH - text_width) / 2, 6)
        else:
            # 标准模式：显示图标和名称
            title = QGraphicsTextItem(f"{self.node.icon} {tr_(self.node.display_name)}", self)
            title.setDefaultTextColor(QColor("#cdd6f4"))
            title.setFont(QFont("Microsoft YaHei", 10, QFont.Weight.Bold))
            title.setPos(8, 4)
    
    def _create_ports(self):
        """创建端口"""
        rect = self.rect()
        node_width = rect.width()
        
        if self.is_compact:
            # 紧凑模式：端口垂直居中，无标签
            y_center = rect.height() / 2
            
            # 输入端口（左侧）
            for port_name, port in self.node.inputs.items():
                port_item = PortItem(port_name, True, Styles.COLORS['green'], self)
                port_item.setPos(0, y_center)
                self.input_ports[port_name] = port_item
            
            # 输出端口（右侧）
            for port_name, port in self.node.outputs.items():
                port_item = PortItem(port_name, False, Styles.COLORS['blue'], self)
                port_item.setPos(node_width, y_center)
                self.output_ports[port_name] = port_item
        else:
            # 标准模式：带标签的端口
            y_offset = self.HEADER_HEIGHT + self.PORT_MARGIN
            
            # 输入端口（左侧）
            for i, (port_name, port) in enumerate(self.node.inputs.items()):
                port_item = PortItem(port_name, True, Styles.COLORS['green'], self)
                port_item.setPos(0, y_offset + i * self.PORT_SPACING)
                self.input_ports[port_name] = port_item
                
                # 端口标签
                label = QGraphicsTextItem(tr_(port.display_name), self)
                label.setDefaultTextColor(QColor("#a6adc8"))
                label.setFont(QFont("Microsoft YaHei", 9))
                label.setPos(12, y_offset + i * self.PORT_SPACING - 8)
            
            # 输出端口（右侧）
            for i, (port_name, port) in enumerate(self.node.outputs.items()):
                port_item = PortItem(port_name, False, Styles.COLORS['blue'], self)
                port_item.setPos(node_width, y_offset + i * self.PORT_SPACING)
                self.output_ports[port_name] = port_item
                
                # 端口标签
                label = QGraphicsTextItem(tr_(port.display_name), self)
                label.setDefaultTextColor(QColor("#a6adc8"))
                label.setFont(QFont("Microsoft YaHei", 9))
                # 右对齐
                text_width = label.boundingRect().width()
                label.setPos(node_width - text_width - 12, y_offset + i * self.PORT_SPACING - 8)
    
    def paint(self, painter: QPainter, option, widget=None):
        """绘制节点"""
        rect = self.rect()
        
        if self.is_compact:
            # 紧凑模式：整体使用分类颜色
            painter.setBrush(QBrush(QColor(self.category_color)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(rect, 8, 8)
        else:
            # 标准模式：背景 + 头部
            # 背景
            painter.setBrush(QBrush(QColor("#313244")))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(rect, 8, 8)
            
            # 头部
            header_rect = QRectF(0, 0, rect.width(), self.HEADER_HEIGHT)
            painter.setBrush(QBrush(QColor(self.category_color)))
            path = QPainterPath()
            path.addRoundedRect(header_rect, 8, 8)
            # 裁剪底部圆角
            clip_rect = QRectF(0, 8, rect.width(), self.HEADER_HEIGHT)
            path.addRect(clip_rect)
            painter.drawPath(path.simplified())
        
        # 执行状态边框
        state_color = Styles.NODE_STATE_COLORS.get(self._execution_state)
        if state_color:
            painter.setBrush(Qt.BrushStyle.NoBrush)
            pen_width = 3 if self._execution_state == 'running' else 2
            painter.setPen(QPen(QColor(state_color), pen_width))
            painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 8, 8)
        
        # 选中边框（优先级最高）
        if self.isSelected():
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor("#f5e0dc"), 2))
            painter.drawRoundedRect(rect, 8, 8)
    
    def set_execution_state(self, state: str):
        """
        设置节点执行状态
        
        Args:
            state: 状态名称 (idle, running, completed, error, waiting)
        """
        self._execution_state = state
        
        # 更新状态图标
        self._update_state_icon()
        
        # 触发重绘
        self.update()
    
    def get_execution_state(self) -> str:
        """获取当前执行状态"""
        return self._execution_state
    
    def _update_state_icon(self):
        """更新状态图标"""
        # 移除旧图标
        if self._state_icon_item:
            self.node_scene.removeItem(self._state_icon_item)
            self._state_icon_item = None
        
        # 获取状态图标
        icon = Styles.NODE_STATE_ICONS.get(self._execution_state, '')
        if not icon:
            return
        
        # 创建图标文本项
        self._state_icon_item = QGraphicsTextItem(icon, self)
        self._state_icon_item.setFont(QFont("Segoe UI Emoji", 10))
        
        # 定位到右上角
        rect = self.rect()
        icon_width = self._state_icon_item.boundingRect().width()
        self._state_icon_item.setPos(rect.width() - icon_width - 2, -2)
    
    def itemChange(self, change, value):
        """位置变化时更新连接线"""
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            # 更新节点位置
            pos = self.pos()
            self.node.position = (pos.x(), pos.y())
            
            # 更新连接线
            for port in list(self.input_ports.values()) + list(self.output_ports.values()):
                for conn in port.connections:
                    conn.update_path()
            
            # 通知场景
            self.node_scene.workflow_changed.emit()
        
        return super().itemChange(change, value)
    
    def mouseDoubleClickEvent(self, event):
        """双击事件"""
        self.node_scene.node_double_clicked.emit(self.node.node_id)
        super().mouseDoubleClickEvent(event)


class ConnectionItem(BaseConnectionItem):
    """
    工作流节点连接线图形项
    
    继承自 BaseConnectionItem，增加对源/目标节点的引用。
    """
    
    def __init__(self, source_port: PortItem, target_port: PortItem, 
                 source_node: 'NodeItem', target_node: 'NodeItem'):
        self.source_node = source_node
        self.target_node = target_node
        super().__init__(source_port, target_port)


# ===== 场景和视图 =====

class NodeGraphScene(BaseGraphScene):
    """
    节点场景
    
    继承自 BaseGraphScene，增加工作流节点特定功能。
    """
    
    node_selected = pyqtSignal(str)
    node_double_clicked = pyqtSignal(str)
    connection_created = pyqtSignal(str, str, str, str)
    workflow_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.node_items: Dict[str, NodeItem] = {}
        self.connection_items: List[ConnectionItem] = []
    
    def add_node_item(self, node: BaseNode) -> NodeItem:
        """添加节点图形项"""
        item = NodeItem(node, self)
        self.addItem(item)
        self.node_items[node.node_id] = item
        return item
    
    def remove_node_item(self, node_id: str):
        """移除节点图形项"""
        if node_id in self.node_items:
            item = self.node_items.pop(node_id)
            
            # 移除相关连接
            for port in list(item.input_ports.values()) + list(item.output_ports.values()):
                for conn in list(port.connections):
                    self.remove_connection_item(conn)
            
            self.removeItem(item)
    
    def reset_node_connections(self, node_id: str):
        """重置节点连接（移除该节点的所有连接）"""
        if node_id in self.node_items:
            node_item = self.node_items[node_id]
            # 先收集所有连接，再逐一删除
            connections_to_remove = []
            for port in list(node_item.input_ports.values()) + list(node_item.output_ports.values()):
                connections_to_remove.extend(list(port.connections))
            
            for conn in connections_to_remove:
                self.remove_connection_item(conn)
            
            node_item.update()

    def add_connection_item(self, source_node_id: str, source_port: str,
                           target_node_id: str, target_port: str) -> Optional[ConnectionItem]:
        """添加连接线"""
        source_node = self.node_items.get(source_node_id)
        target_node = self.node_items.get(target_node_id)
        
        if not source_node or not target_node:
            return None
        
        source_port_item = source_node.output_ports.get(source_port)
        target_port_item = target_node.input_ports.get(target_port)
        
        if not source_port_item or not target_port_item:
            return None
        
        conn = ConnectionItem(source_port_item, target_port_item, source_node, target_node)
        self.addItem(conn)
        self.connection_items.append(conn)
        return conn
    
    def remove_connection_item(self, conn: ConnectionItem):
        """移除连接线"""
        if conn in self.connection_items:
            self.connection_items.remove(conn)
        conn.remove()
        self.removeItem(conn)
    
    def clear_all(self):
        """清空所有"""
        for conn in list(self.connection_items):
            self.remove_connection_item(conn)
        for node_id in list(self.node_items.keys()):
            self.remove_node_item(node_id)
    
    def update_node_state(self, node_id: str, state: str):
        """
        更新节点的执行状态
        
        Args:
            node_id: 节点ID
            state: 状态名称 (idle, running, completed, error, waiting)
        """
        if node_id in self.node_items:
            self.node_items[node_id].set_execution_state(state)
    
    def reset_all_node_states(self):
        """重置所有节点的执行状态为 idle"""
        for node_item in self.node_items.values():
            node_item.set_execution_state('idle')
    
    def highlight_node(self, node_id: str):
        """高亮显示指定节点（居中并选中）"""
        if node_id in self.node_items:
            node_item = self.node_items[node_id]
            # 清除其他选中
            self.clearSelection()
            # 选中目标节点
            node_item.setSelected(True)
            # 居中显示
            if self.views():
                self.views()[0].centerOn(node_item)
    
    # drawBackground 继承自 BaseGraphScene
    
    def mousePressEvent(self, event):
        """鼠标按下事件"""
        item = self.itemAt(event.scenePos(), self.views()[0].transform() if self.views() else None)
        
        # 检查是否点击了端口
        if isinstance(item, PortItem):
            self.start_connection_drawing(item, event.scenePos())
            return
        
        # 检查是否选中了节点
        if isinstance(item, NodeItem) or (item and isinstance(item.parentItem(), NodeItem)):
            node_item = item if isinstance(item, NodeItem) else item.parentItem()
            self.node_selected.emit(node_item.node.node_id)
        
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """鼠标移动事件"""
        self.update_connection_drawing(event.scenePos())
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        if self.is_drawing_connection:
            self._finish_connection(event.scenePos())
        super().mouseReleaseEvent(event)
    
    def _finish_connection(self, pos: QPointF):
        """完成连接"""
        start_port, start_node = self.end_connection_drawing()
        
        if not start_port:
            return
        
        # 查找目标端口
        item = self.itemAt(pos, self.views()[0].transform() if self.views() else None)
        
        if isinstance(item, PortItem) and item != start_port:
            # 确定源和目标
            if start_port.is_input and not item.is_input:
                # 从输入拖到输出，反转
                source_port = item
                target_port = start_port
                source_node = item.parentItem()
                target_node = start_node
            elif not start_port.is_input and item.is_input:
                # 从输出拖到输入
                source_port = start_port
                target_port = item
                source_node = start_node
                target_node = item.parentItem()
            else:
                # 同类型端口，无效
                return
            
            # 发送连接信号
            self.connection_created.emit(
                source_node.node.node_id,
                source_port.port_name,
                target_node.node.node_id,
                target_port.port_name
            )


class NodeGraphView(BaseGraphView):
    """
    节点视图
    
    继承自 BaseGraphView，增加工作流节点拖放功能。
    """
    
    def __init__(self, scene: NodeGraphScene, parent=None):
        super().__init__(scene, parent)
    
    # wheelEvent, mousePressEvent, mouseMoveEvent, mouseReleaseEvent, keyPressEvent
    # 继承自 BaseGraphView
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """拖入事件"""
        if event.mimeData().hasFormat("application/x-workflow-node"):
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            event.ignore()
    
    def dragMoveEvent(self, event):
        """拖拽移动事件 - 持续接受拖拽"""
        if event.mimeData().hasFormat("application/x-workflow-node"):
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            event.ignore()
    
    def dropEvent(self, event: QDropEvent):
        """放置事件"""
        if event.mimeData().hasFormat("application/x-workflow-node"):
            node_type = bytes(event.mimeData().data("application/x-workflow-node")).decode()
            # PyQt6 使用 position() 代替 pos()
            pos = self.mapToScene(event.position().toPoint())
            
            # 获取父组件并添加节点
            parent = self.parent()
            if hasattr(parent, 'add_node'):
                parent.add_node(node_type, (pos.x(), pos.y()))
            
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            event.ignore()


# ===== 主组件 =====

class NodeGraphWidget(QWidget):
    """
    节点画布组件
    
    Signals:
        node_selected: 节点被选中 (node_id)
        node_double_clicked: 节点被双击 (node_id)
        connection_created: 连接被创建 (source_node_id, source_port, target_node_id, target_port)
        workflow_changed: 工作流发生变化
    """
    
    node_selected = pyqtSignal(str)
    node_double_clicked = pyqtSignal(str)
    connection_created = pyqtSignal(str, str, str, str)
    workflow_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.workflow: Optional[Workflow] = None
        
        self._setup_ui()
        self._connect_signals()
    
    def _setup_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建场景和视图
        self._scene = NodeGraphScene()
        self._view = NodeGraphView(self._scene, self)
        layout.addWidget(self._view)
        
        # 设置右键菜单
        self._view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._view.customContextMenuRequested.connect(self._show_context_menu)
    
    def _connect_signals(self):
        """连接信号"""
        self._scene.node_selected.connect(self.node_selected)
        self._scene.node_double_clicked.connect(self.node_double_clicked)
        self._scene.connection_created.connect(self._on_connection_created)
        self._scene.workflow_changed.connect(self.workflow_changed)
    
    def set_workflow(self, workflow: Workflow):
        """设置工作流"""
        self.workflow = workflow
        self._sync_from_workflow()
    
    def get_workflow(self) -> Optional[Workflow]:
        """获取工作流"""
        return self.workflow
    
    def _sync_from_workflow(self):
        """从工作流同步到画布"""
        if not self.workflow:
            return
        
        # 清空画布
        self._scene.clear_all()
        
        # 添加节点
        for node_id, node in self.workflow.nodes.items():
            self._scene.add_node_item(node)
        
        # 添加连接
        for conn in self.workflow.connections:
            self._scene.add_connection_item(
                conn.source_node_id,
                conn.source_port,
                conn.target_node_id,
                conn.target_port
            )
    
    def add_node(self, node_type: str, position: tuple = None) -> Optional[str]:
        """添加节点"""
        if not self.workflow:
            self.workflow = Workflow()
        
        if position is None:
            position = (0, 0)
        
        # 创建节点
        node = self.workflow.create_and_add_node(node_type, position)
        if not node:
            return None
        
        # 添加到画布
        self._scene.add_node_item(node)
        
        self.workflow_changed.emit()
        return node.node_id
    
    def remove_node(self, node_id: str):
        """移除节点"""
        if not self.workflow:
            return
        
        # 从画布移除
        self._scene.remove_node_item(node_id)
        
        # 从工作流移除
        self.workflow.remove_node(node_id)
        
        self.workflow_changed.emit()

    def _on_connection_created(self, source_node_id: str, source_port: str,
                               target_node_id: str, target_port: str):
        """连接创建处理"""
        if not self.workflow:
            return
        
        # 添加到工作流
        success, msg = self.workflow.connect(
            source_node_id, source_port,
            target_node_id, target_port
        )
        
        if success:
            # 添加到画布
            self._scene.add_connection_item(
                source_node_id, source_port,
                target_node_id, target_port
            )
            self.connection_created.emit(
                source_node_id, source_port,
                target_node_id, target_port
            )
            self.workflow_changed.emit()
        else:
            logger.warning(f"创建连接失败: {msg}")
            # 显示提示信息
            from PyQt6.QtWidgets import QToolTip, QMessageBox
            QMessageBox.warning(self, tr_("Connection failed"), msg)
    
    def get_selected_node_id(self) -> Optional[str]:
        """获取选中的节点ID"""
        selected = self._scene.selectedItems()
        for item in selected:
            if isinstance(item, NodeItem):
                return item.node.node_id
        return None
    
    def _show_context_menu(self, pos):
        """显示右键菜单"""
        menu = QMenu(self)
        
        # 注：新增节点统一从左侧「节点库」添加，避免右键菜单随节点数量膨胀。
        
        # 重置选中节点连接
        reset_conn_action = menu.addAction(tr_("Reset selected node connections"))
        reset_conn_action.triggered.connect(self._reset_selected_connections)
        
        # 删除选中节点
        delete_action = menu.addAction(tr_("Delete selected nodes"))
        delete_action.triggered.connect(self._delete_selected)
        
        menu.exec(self._view.mapToGlobal(pos))
    
    def _delete_selected(self):
        """Delete selected items.

        Priority:
        - Delete selected connection(s) if any
        - Otherwise delete selected node(s)
        """
        selected = list(self._scene.selectedItems())

        # 1) Delete selected connections first
        conn_items = [item for item in selected if isinstance(item, ConnectionItem)]
        if conn_items:
            for conn_item in list(conn_items):
                if self.workflow:
                    connection = Connection(
                        source_node_id=conn_item.source_node.node.node_id,
                        source_port=conn_item.source_port.port_name,
                        target_node_id=conn_item.target_node.node.node_id,
                        target_port=conn_item.target_port.port_name,
                    )
                    self.workflow.remove_connection(connection)
                self._scene.remove_connection_item(conn_item)

            self.workflow_changed.emit()
            return

        # 2) Delete selected nodes
        node_ids: List[str] = []
        for item in selected:
            if isinstance(item, NodeItem):
                node_ids.append(item.node.node_id)

        for node_id in node_ids:
            self.remove_node(node_id)
    
    def _reset_selected_connections(self):
        """重置所有选中节点的连接"""
        selected = self._scene.selectedItems()
        node_ids = []
        for item in selected:
            if isinstance(item, NodeItem):
                node_ids.append(item.node.node_id)
        
        if not node_ids:
            return
        
        # 重置每个选中节点的连接
        for node_id in node_ids:
            self._scene.reset_node_connections(node_id)
            # 同步到工作流
            if self.workflow:
                self.workflow.disconnect_node(node_id)
        
        self.workflow_changed.emit()
    
    def clear(self):
        """清空画布"""
        self._scene.clear_all()
        self.workflow = None
    
    def fit_to_selection(self):
        """适应选中内容"""
        self._view.fitInView(self._scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
    
    def center_on(self, node_id: str):
        """居中显示指定节点"""
        if node_id in self._scene.node_items:
            item = self._scene.node_items[node_id]
            self._view.centerOn(item)
    
    # ===== 节点执行状态管理 =====
    
    def update_node_state(self, node_id: str, state: str):
        """
        更新节点的执行状态
        
        Args:
            node_id: 节点ID
            state: 状态名称 (idle, running, completed, error, waiting)
        """
        self._scene.update_node_state(node_id, state)
    
    def reset_all_node_states(self):
        """重置所有节点的执行状态为 idle"""
        self._scene.reset_all_node_states()
    
    def highlight_node(self, node_id: str):
        """高亮显示指定节点（居中并选中）"""
        self._scene.highlight_node(node_id)
