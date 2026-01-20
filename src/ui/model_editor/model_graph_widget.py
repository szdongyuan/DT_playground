# -*- coding: utf-8 -*-
"""
模型画布组件

可视化编辑神经网络结构的画布。
"""

import logging
from typing import Dict, List, Optional

from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QBrush, QColor, QDragEnterEvent, QDropEvent, QFont,
    QKeyEvent, QMouseEvent, QPainter, QPainterPath, QPen, QWheelEvent
)
from PyQt6.QtWidgets import (
    QGraphicsEllipseItem, QGraphicsItem, QGraphicsLineItem,
    QGraphicsPathItem, QGraphicsRectItem, QGraphicsScene,
    QGraphicsTextItem, QGraphicsView, QMenu, QVBoxLayout, QWidget
)

from src.model_builder.layer_base import (
    create_layer, get_layer_class, get_layers_by_category,
    LayerCategory, LayerNode
)
from src.model_builder.model_graph import ModelConnection, ModelGraph
from src.ui.styles import Styles

logger = logging.getLogger(__name__)


class LayerPortItem(QGraphicsEllipseItem):
    """层端口图形项"""
    
    PORT_RADIUS = 6
    
    def __init__(self, is_input: bool, color: str, parent=None):
        super().__init__(parent)
        self.is_input = is_input
        self.connections: List['LayerConnectionItem'] = []
        
        self.setRect(-self.PORT_RADIUS, -self.PORT_RADIUS,
                     self.PORT_RADIUS * 2, self.PORT_RADIUS * 2)
        self.setBrush(QBrush(QColor(color)))
        self.setPen(QPen(QColor("#1e1e2e"), 2))
        self.setAcceptHoverEvents(True)
    
    def hoverEnterEvent(self, event):
        self.setPen(QPen(QColor("#f5e0dc"), 3))
        super().hoverEnterEvent(event)
    
    def hoverLeaveEvent(self, event):
        self.setPen(QPen(QColor("#1e1e2e"), 2))
        super().hoverLeaveEvent(event)


class LayerItem(QGraphicsRectItem):
    """层图形项"""
    
    LAYER_WIDTH = 180
    LAYER_HEIGHT = 60
    HEADER_HEIGHT = 28
    
    def __init__(self, layer: LayerNode, scene: 'ModelGraphScene'):
        super().__init__()
        self.layer = layer
        self.layer_scene = scene
        
        # 设置矩形
        self.setRect(0, 0, self.LAYER_WIDTH, self.LAYER_HEIGHT)
        
        # 设置属性
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)
        
        # 设置位置
        self.setPos(layer.position[0], layer.position[1])
        
        # 颜色
        self.category_color = layer.category.color
        
        # 创建子元素
        self._create_header()
        self._create_ports()
    
    def _create_header(self):
        """创建头部"""
        title = QGraphicsTextItem(f"{self.layer.icon} {self.layer.display_name}", self)
        title.setDefaultTextColor(QColor("#cdd6f4"))
        title.setFont(QFont("Microsoft YaHei", 10, QFont.Weight.Bold))
        title.setPos(8, 4)
    
    def _create_ports(self):
        """创建端口"""
        # 输入端口（左侧中央）
        self.input_port = LayerPortItem(True, Styles.COLORS['green'], self)
        self.input_port.setPos(0, self.LAYER_HEIGHT / 2)
        
        # 输出端口（右侧中央）
        self.output_port = LayerPortItem(False, Styles.COLORS['blue'], self)
        self.output_port.setPos(self.LAYER_WIDTH, self.LAYER_HEIGHT / 2)
    
    def paint(self, painter: QPainter, option, widget=None):
        """绘制层"""
        rect = self.rect()
        
        # 背景
        painter.setBrush(QBrush(QColor("#313244")))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(rect, 8, 8)
        
        # 头部颜色（如果冻结则使用灰色调）
        header_color = self.category_color
        if not self.layer.trainable:
            # 冻结状态：使用较暗的颜色
            header_color = "#6c7086"  # Catppuccin overlay0
        
        header_rect = QRectF(0, 0, rect.width(), self.HEADER_HEIGHT)
        painter.setBrush(QBrush(QColor(header_color)))
        path = QPainterPath()
        path.addRoundedRect(header_rect, 8, 8)
        clip_rect = QRectF(0, 8, rect.width(), self.HEADER_HEIGHT)
        path.addRect(clip_rect)
        painter.drawPath(path.simplified())
        
        # 绘制状态图标（右上角）
        self._draw_status_icons(painter, rect)
        
        # 选中边框
        if self.isSelected():
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor("#f5e0dc"), 2))
            painter.drawRoundedRect(rect, 8, 8)
    
    def _draw_status_icons(self, painter: QPainter, rect: QRectF):
        """绘制层状态图标（权重和冻结状态）"""
        icons = []
        
        # 权重状态
        if self.layer.has_weights:
            icons.append("✓")
        
        # 冻结状态
        if not self.layer.trainable:
            icons.append("🔒")
        
        if icons:
            status_text = " ".join(icons)
            painter.setPen(QPen(QColor("#cdd6f4")))
            painter.setFont(QFont("Segoe UI Emoji", 9))
            
            # 在底部区域绘制状态
            status_rect = QRectF(
                rect.width() - 50, 
                self.HEADER_HEIGHT + 4,
                45,
                rect.height() - self.HEADER_HEIGHT - 8
            )
            painter.drawText(status_rect, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, status_text)
    
    def itemChange(self, change, value):
        """位置变化时更新连接线"""
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            pos = self.pos()
            self.layer.position = (pos.x(), pos.y())
            
            # 更新连接线（确保端口已创建）
            if hasattr(self, 'input_port') and hasattr(self, 'output_port'):
                for conn in self.input_port.connections + self.output_port.connections:
                    conn.update_path()
                
                self.layer_scene.graph_changed.emit()
        
        return super().itemChange(change, value)
    
    def mouseDoubleClickEvent(self, event):
        """双击事件"""
        self.layer_scene.layer_double_clicked.emit(self.layer.layer_id)
        super().mouseDoubleClickEvent(event)


class LayerConnectionItem(QGraphicsPathItem):
    """层连接线图形项"""
    
    def __init__(self, source_layer: LayerItem, target_layer: LayerItem):
        super().__init__()
        self.source_layer = source_layer
        self.target_layer = target_layer
        
        # 添加到端口的连接列表
        source_layer.output_port.connections.append(self)
        target_layer.input_port.connections.append(self)
        
        # 样式
        self.setPen(QPen(QColor(Styles.COLORS['blue']), 2))
        self.setZValue(-1)
        
        self.update_path()
    
    def update_path(self):
        """更新路径"""
        start = self.source_layer.output_port.scenePos()
        end = self.target_layer.input_port.scenePos()
        
        path = QPainterPath()
        path.moveTo(start)
        
        dx = abs(end.x() - start.x()) / 2
        ctrl1 = QPointF(start.x() + dx, start.y())
        ctrl2 = QPointF(end.x() - dx, end.y())
        
        path.cubicTo(ctrl1, ctrl2, end)
        self.setPath(path)
    
    def remove(self):
        """移除连接"""
        if self in self.source_layer.output_port.connections:
            self.source_layer.output_port.connections.remove(self)
        if self in self.target_layer.input_port.connections:
            self.target_layer.input_port.connections.remove(self)


class ModelGraphScene(QGraphicsScene):
    """模型场景"""
    
    layer_selected = pyqtSignal(str)
    layer_double_clicked = pyqtSignal(str)
    connection_created = pyqtSignal(str, str)
    graph_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSceneRect(-5000, -5000, 10000, 10000)
        self.setBackgroundBrush(QBrush(QColor(Styles.COLORS['base'])))
        
        self.layer_items: Dict[str, LayerItem] = {}
        self.connection_items: List[LayerConnectionItem] = []
        
        # 连接绘制状态
        self._drawing_connection = False
        self._temp_line: Optional[QGraphicsLineItem] = None
        self._start_port: Optional[LayerPortItem] = None
        self._start_layer: Optional[LayerItem] = None
    
    def add_layer_item(self, layer: LayerNode) -> LayerItem:
        """添加层图形项"""
        item = LayerItem(layer, self)
        self.addItem(item)
        self.layer_items[layer.layer_id] = item
        return item
    
    def remove_layer_item(self, layer_id: str):
        """移除层图形项"""
        if layer_id in self.layer_items:
            item = self.layer_items.pop(layer_id)
            
            # 移除相关连接
            for conn in list(item.input_port.connections) + list(item.output_port.connections):
                self.remove_connection_item(conn)
            
            self.removeItem(item)
    
    def add_connection_item(self, source_layer_id: str, target_layer_id: str) -> Optional[LayerConnectionItem]:
        """添加连接线"""
        source_layer = self.layer_items.get(source_layer_id)
        target_layer = self.layer_items.get(target_layer_id)
        
        if not source_layer or not target_layer:
            return None
        
        conn = LayerConnectionItem(source_layer, target_layer)
        self.addItem(conn)
        self.connection_items.append(conn)
        return conn
    
    def remove_connection_item(self, conn: LayerConnectionItem):
        """移除连接线"""
        if conn in self.connection_items:
            self.connection_items.remove(conn)
        conn.remove()
        self.removeItem(conn)
    
    def reset_layer_connections(self, layer_id: str):
        """重置层连接（移除该层的所有连接）"""
        if layer_id in self.layer_items:
            layer_item = self.layer_items[layer_id]
            # 先收集所有连接，再逐一删除
            connections_to_remove = list(layer_item.input_port.connections) + list(layer_item.output_port.connections)
            
            for conn in connections_to_remove:
                self.remove_connection_item(conn)
            
            layer_item.update()
    
    def clear_all(self):
        """清空所有"""
        for conn in list(self.connection_items):
            self.remove_connection_item(conn)
        for layer_id in list(self.layer_items.keys()):
            self.remove_layer_item(layer_id)
    
    def drawBackground(self, painter: QPainter, rect: QRectF):
        """绘制网格背景"""
        super().drawBackground(painter, rect)
        
        grid_size = 20
        left = int(rect.left()) - (int(rect.left()) % grid_size)
        top = int(rect.top()) - (int(rect.top()) % grid_size)
        
        lines = []
        for x in range(left, int(rect.right()), grid_size):
            lines.append(((x, rect.top()), (x, rect.bottom())))
        for y in range(top, int(rect.bottom()), grid_size):
            lines.append(((rect.left(), y), (rect.right(), y)))
        
        painter.setPen(QPen(QColor("#252535"), 1))
        for line in lines:
            painter.drawLine(QPointF(*line[0]), QPointF(*line[1]))
    
    def mousePressEvent(self, event):
        """鼠标按下事件"""
        item = self.itemAt(event.scenePos(), self.views()[0].transform() if self.views() else None)
        
        if isinstance(item, LayerPortItem):
            self._start_connection(item, event.scenePos())
            return
        
        if isinstance(item, LayerItem) or (item and isinstance(item.parentItem(), LayerItem)):
            layer_item = item if isinstance(item, LayerItem) else item.parentItem()
            self.layer_selected.emit(layer_item.layer.layer_id)
        
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """鼠标移动事件"""
        if self._drawing_connection and self._temp_line:
            line = self._temp_line.line()
            self._temp_line.setLine(line.x1(), line.y1(),
                                    event.scenePos().x(), event.scenePos().y())
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        if self._drawing_connection:
            self._finish_connection(event.scenePos())
        super().mouseReleaseEvent(event)
    
    def _start_connection(self, port: LayerPortItem, pos: QPointF):
        """开始绘制连接"""
        self._drawing_connection = True
        self._start_port = port
        self._start_layer = port.parentItem()
        
        self._temp_line = QGraphicsLineItem()
        self._temp_line.setPen(QPen(QColor(Styles.COLORS['yellow']), 2, Qt.PenStyle.DashLine))
        port_pos = port.scenePos()
        self._temp_line.setLine(port_pos.x(), port_pos.y(), pos.x(), pos.y())
        self.addItem(self._temp_line)
    
    def _finish_connection(self, pos: QPointF):
        """完成连接"""
        if self._temp_line:
            self.removeItem(self._temp_line)
            self._temp_line = None
        
        item = self.itemAt(pos, self.views()[0].transform() if self.views() else None)
        
        if isinstance(item, LayerPortItem) and item != self._start_port:
            # 确定源和目标
            if self._start_port.is_input and not item.is_input:
                source_layer = item.parentItem()
                target_layer = self._start_layer
            elif not self._start_port.is_input and item.is_input:
                source_layer = self._start_layer
                target_layer = item.parentItem()
            else:
                self._drawing_connection = False
                self._start_port = None
                self._start_layer = None
                return
            
            self.connection_created.emit(
                source_layer.layer.layer_id,
                target_layer.layer.layer_id
            )
        
        self._drawing_connection = False
        self._start_port = None
        self._start_layer = None


class ModelGraphView(QGraphicsView):
    """模型视图"""
    
    def __init__(self, scene: ModelGraphScene, parent=None):
        super().__init__(scene, parent)
        
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setAcceptDrops(True)
        
        self._is_panning = False
        self._last_pan_pos = QPoint()
    
    def wheelEvent(self, event: QWheelEvent):
        """缩放"""
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)
    
    def mousePressEvent(self, event: QMouseEvent):
        """鼠标按下"""
        if event.button() == Qt.MouseButton.MiddleButton:
            self._is_panning = True
            self._last_pan_pos = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        else:
            super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """鼠标移动"""
        if self._is_panning:
            delta = event.pos() - self._last_pan_pos
            self._last_pan_pos = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
        else:
            super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """鼠标释放"""
        if event.button() == Qt.MouseButton.MiddleButton:
            self._is_panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            super().mouseReleaseEvent(event)
    
    def keyPressEvent(self, event: QKeyEvent):
        """键盘按下事件 - 支持 DEL 键删除选中层"""
        if event.key() == Qt.Key.Key_Delete or event.key() == Qt.Key.Key_Backspace:
            parent = self.parent()
            if hasattr(parent, '_delete_selected'):
                parent._delete_selected()
            return
        super().keyPressEvent(event)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """拖入事件"""
        if event.mimeData().hasFormat("application/x-model-layer"):
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            event.ignore()
    
    def dragMoveEvent(self, event):
        """拖拽移动事件 - 持续接受拖拽"""
        if event.mimeData().hasFormat("application/x-model-layer"):
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            event.ignore()
    
    def dropEvent(self, event: QDropEvent):
        """放置事件"""
        if event.mimeData().hasFormat("application/x-model-layer"):
            layer_type = bytes(event.mimeData().data("application/x-model-layer")).decode()
            # PyQt6 使用 position() 代替 pos()
            pos = self.mapToScene(event.position().toPoint())
            
            parent = self.parent()
            if hasattr(parent, 'add_layer'):
                parent.add_layer(layer_type, (pos.x(), pos.y()))
            
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            event.ignore()


class ModelGraphWidget(QWidget):
    """
    模型画布组件
    """
    
    layer_selected = pyqtSignal(str)
    layer_double_clicked = pyqtSignal(str)
    connection_created = pyqtSignal(str, str)
    graph_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.model_graph: Optional[ModelGraph] = None
        
        self._setup_ui()
        self._connect_signals()
    
    def _setup_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self._scene = ModelGraphScene()
        self._view = ModelGraphView(self._scene, self)
        layout.addWidget(self._view)
        
        self._view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._view.customContextMenuRequested.connect(self._show_context_menu)
    
    def _connect_signals(self):
        """连接信号"""
        self._scene.layer_selected.connect(self.layer_selected)
        self._scene.layer_double_clicked.connect(self.layer_double_clicked)
        self._scene.connection_created.connect(self._on_connection_created)
        self._scene.graph_changed.connect(self.graph_changed)
    
    def set_model_graph(self, graph: ModelGraph):
        """设置模型图"""
        self.model_graph = graph
        self._sync_from_graph()
    
    def get_model_graph(self) -> Optional[ModelGraph]:
        """获取模型图"""
        return self.model_graph
    
    def _sync_from_graph(self):
        """从模型图同步到画布"""
        if not self.model_graph:
            return
        
        self._scene.clear_all()
        
        for layer_id, layer in self.model_graph.layers.items():
            self._scene.add_layer_item(layer)
        
        for conn in self.model_graph.connections:
            self._scene.add_connection_item(
                conn.source_layer_id,
                conn.target_layer_id
            )
    
    def add_layer(self, layer_type: str, position: tuple = None) -> Optional[str]:
        """添加层"""
        if not self.model_graph:
            self.model_graph = ModelGraph()
        
        if position is None:
            position = (0, 0)
        
        layer = self.model_graph.create_and_add_layer(layer_type, position)
        if not layer:
            return None
        
        self._scene.add_layer_item(layer)
        self.graph_changed.emit()
        return layer.layer_id
    
    def remove_layer(self, layer_id: str):
        """移除层"""
        if not self.model_graph:
            return
        
        self._scene.remove_layer_item(layer_id)
        self.model_graph.remove_layer(layer_id)
        self.graph_changed.emit()
    
    def _on_connection_created(self, source_layer_id: str, target_layer_id: str):
        """连接创建处理"""
        if not self.model_graph:
            return
        
        success, msg = self.model_graph.connect(source_layer_id, target_layer_id)
        
        if success:
            self._scene.add_connection_item(source_layer_id, target_layer_id)
            self.connection_created.emit(source_layer_id, target_layer_id)
            self.graph_changed.emit()
        else:
            logger.warning(f"创建连接失败: {msg}")
    
    def get_selected_layer_id(self) -> Optional[str]:
        """获取选中的层ID"""
        selected = self._scene.selectedItems()
        for item in selected:
            if isinstance(item, LayerItem):
                return item.layer.layer_id
        return None
    
    def get_selected_layer_ids(self) -> List[str]:
        """获取所有选中的层ID"""
        selected = self._scene.selectedItems()
        layer_ids = []
        for item in selected:
            if isinstance(item, LayerItem):
                layer_ids.append(item.layer.layer_id)
        return layer_ids
    
    def _show_context_menu(self, pos):
        """显示右键菜单"""
        menu = QMenu(self)
        scene_pos = self._view.mapToScene(pos)
        
        add_menu = menu.addMenu("添加层")
        
        for category in LayerCategory:
            layer_classes = get_layers_by_category(category)
            if not layer_classes:
                continue
            
            category_menu = add_menu.addMenu(f"{category.display_name}")
            
            for layer_class in layer_classes:
                action = category_menu.addAction(
                    f"{layer_class.icon} {layer_class.display_name}"
                )
                action.setData(layer_class.layer_type)
                action.triggered.connect(
                    lambda checked, lt=layer_class.layer_type, p=(scene_pos.x(), scene_pos.y()):
                        self.add_layer(lt, p)
                )
        
        menu.addSeparator()
        
        # 重置选中层连接
        reset_conn_action = menu.addAction("重置选中层连接")
        reset_conn_action.triggered.connect(self._reset_selected_connections)
        
        delete_action = menu.addAction("删除选中层")
        delete_action.triggered.connect(self._delete_selected)
        
        menu.exec(self._view.mapToGlobal(pos))
    
    def _delete_selected(self):
        """删除选中的层（支持多选）"""
        layer_ids = self.get_selected_layer_ids()
        for layer_id in layer_ids:
            self.remove_layer(layer_id)
    
    def _reset_selected_connections(self):
        """重置所有选中层的连接"""
        layer_ids = self.get_selected_layer_ids()
        
        if not layer_ids:
            return
        
        # 重置每个选中层的连接
        for layer_id in layer_ids:
            self._scene.reset_layer_connections(layer_id)
            # 同步到模型图
            if self.model_graph:
                self.model_graph.disconnect_layer(layer_id)
        
        self.graph_changed.emit()
    
    def clear(self):
        """清空画布"""
        self._scene.clear_all()
        self.model_graph = None
    
    def fit_to_selection(self):
        """适应选中内容"""
        self._view.fitInView(self._scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
    
    def center_on(self, layer_id: str):
        """居中显示指定层"""
        if layer_id in self._scene.layer_items:
            item = self._scene.layer_items[layer_id]
            self._view.centerOn(item)

