# -*- coding: utf-8 -*-
"""
Graph Editor Base Classes

Provides reusable graphics item base classes for node_editor and model_editor.
"""

import logging
from typing import Iterable, List, Optional

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush, QColor, QFont, QKeyEvent, QKeySequence, QMouseEvent, QPainter,
    QPainterPath, QPen, QWheelEvent
)
from PySide6.QtWidgets import (
    QGraphicsEllipseItem, QGraphicsItem, QGraphicsLineItem,
    QGraphicsPathItem, QGraphicsRectItem, QGraphicsScene,
    QGraphicsView
)

from ..styles import Styles


logger = logging.getLogger(__name__)


class BasePortItem(QGraphicsEllipseItem):
    """
    端口图形项基类
    
    为工作流节点端口和模型层端口提供通用的可视化和交互功能。
    
    子类可覆盖:
    - PORT_RADIUS: 端口半径
    - HOVER_COLOR: 悬停时的边框颜色
    - BORDER_COLOR: 默认边框颜色
    """
    
    PORT_RADIUS = 6
    HOVER_COLOR = "#f5e0dc"
    BORDER_COLOR = "#1e1e2e"
    
    def __init__(self, is_input: bool, color: str, parent=None):
        """
        初始化端口项
        
        Args:
            is_input: 是否为输入端口
            color: 端口填充颜色
            parent: 父图形项
        """
        super().__init__(parent)
        self.is_input = is_input
        self.color = color
        self.connections: List['BaseConnectionItem'] = []
        
        # 设置样式
        self.setRect(
            -self.PORT_RADIUS, -self.PORT_RADIUS,
            self.PORT_RADIUS * 2, self.PORT_RADIUS * 2
        )
        self.setBrush(QBrush(QColor(color)))
        self.setPen(QPen(QColor(self.BORDER_COLOR), 2))
        
        # 允许交互
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
    
    def hoverEnterEvent(self, event):
        """悬停进入"""
        self.setPen(QPen(QColor(self.HOVER_COLOR), 3))
        super().hoverEnterEvent(event)
    
    def hoverLeaveEvent(self, event):
        """悬停离开"""
        self.setPen(QPen(QColor(self.BORDER_COLOR), 2))
        super().hoverLeaveEvent(event)


class BaseConnectionItem(QGraphicsPathItem):
    """
    连接线图形项基类
    
    提供连接线的绘制和更新功能。
    """
    
    ORTHOGONAL_EXIT_LENGTH = 32.0
    ORTHOGONAL_CORNER_RADIUS = 8.0
    PARALLEL_ROUTE_SPACING = 10.0
    REVERSE_ROUTE_MARGIN = 48.0
    
    def __init__(self, source_port: BasePortItem, target_port: BasePortItem,
                 line_color: str = None):
        """
        初始化连接线
        
        Args:
            source_port: 源端口
            target_port: 目标端口
            line_color: 连接线颜色（默认使用 Styles.COLORS['blue']）
        """
        super().__init__()
        self.source_port = source_port
        self.target_port = target_port
        
        # 添加到端口的连接列表
        source_port.connections.append(self)
        target_port.connections.append(self)
        
        # 样式
        self._base_color = line_color or Styles.COLORS['blue']
        self._selected_color = "#f5e0dc"
        self._base_width = 2
        self._selected_width = 3

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self._update_pen()
        self.setZValue(-1)  # 在节点下方
        
        self.update_path()

    def _update_pen(self):
        """Update pen according to selection state."""
        if self.isSelected():
            self.setPen(QPen(QColor(self._selected_color), self._selected_width))
        else:
            self.setPen(QPen(QColor(self._base_color), self._base_width))

    def itemChange(self, change, value):
        """React to selection changes."""
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            self._update_pen()
        return super().itemChange(change, value)
    
    def _port_exit_direction(self, port: BasePortItem) -> float:
        """Return the horizontal direction used when leaving a port."""
        return -1.0 if port.is_input else 1.0
    
    def _parallel_route_offset(self) -> float:
        """Return a small trunk offset to keep parallel routes readable."""
        sibling_connections = [
            conn for conn in self.source_port.connections
            if conn.source_port is self.source_port
            and conn.target_port is self.target_port
        ]
        if len(sibling_connections) <= 1:
            return 0.0
        
        index = sibling_connections.index(self)
        center = (len(sibling_connections) - 1) / 2
        return (index - center) * self.PARALLEL_ROUTE_SPACING
    
    def _item_scene_bounds(self, port: BasePortItem) -> QRectF:
        """Return the parent item bounds used for outside routing."""
        parent = port.parentItem()
        if parent is not None:
            return parent.sceneBoundingRect()
        
        pos = port.scenePos()
        return QRectF(pos.x(), pos.y(), 0.0, 0.0)
    
    def _outside_route_y(
        self,
        start: QPointF,
        end: QPointF,
        source_bounds: QRectF,
        target_bounds: QRectF,
        route_offset: float,
    ) -> float:
        """Choose the nearest outside channel above or below both items."""
        if source_bounds.bottom() < target_bounds.top():
            corridor_top = source_bounds.bottom()
            corridor_bottom = target_bounds.top()
            return (corridor_top + corridor_bottom) / 2 + route_offset
        
        if target_bounds.bottom() < source_bounds.top():
            corridor_top = target_bounds.bottom()
            corridor_bottom = source_bounds.top()
            return (corridor_top + corridor_bottom) / 2 + route_offset
        
        top = min(source_bounds.top(), target_bounds.top())
        bottom = max(source_bounds.bottom(), target_bounds.bottom())
        above = top - self.REVERSE_ROUTE_MARGIN
        below = bottom + self.REVERSE_ROUTE_MARGIN
        
        above_distance = abs(start.y() - above) + abs(end.y() - above)
        below_distance = abs(start.y() - below) + abs(end.y() - below)
        route_y = above if above_distance < below_distance else below
        return route_y + route_offset
    
    def _reverse_route_points(
        self,
        start: QPointF,
        end: QPointF,
        route_offset: float,
    ) -> List[QPointF]:
        """Build an outside dogleg route for connections flowing backward."""
        source_bounds = self._item_scene_bounds(self.source_port)
        target_bounds = self._item_scene_bounds(self.target_port)
        source_outer_x = max(
            start.x() + self.ORTHOGONAL_EXIT_LENGTH,
            source_bounds.right() + self.REVERSE_ROUTE_MARGIN,
        )
        target_outer_x = min(
            end.x() - self.ORTHOGONAL_EXIT_LENGTH,
            target_bounds.left() - self.REVERSE_ROUTE_MARGIN,
        )
        route_y = self._outside_route_y(
            start,
            end,
            source_bounds,
            target_bounds,
            route_offset,
        )
        
        return [
            start,
            QPointF(source_outer_x, start.y()),
            QPointF(source_outer_x, route_y),
            QPointF(target_outer_x, route_y),
            QPointF(target_outer_x, end.y()),
            end,
        ]
    
    def _append_rounded_corner_path(
        self,
        path: QPainterPath,
        points: List[QPointF],
    ) -> None:
        """Append a rounded orthogonal polyline to an existing path."""
        compact_points = []
        for point in points:
            if not compact_points or point != compact_points[-1]:
                compact_points.append(point)
        
        if len(compact_points) < 2:
            return
        
        path.moveTo(compact_points[0])
        
        for index in range(1, len(compact_points) - 1):
            previous = compact_points[index - 1]
            current = compact_points[index]
            next_point = compact_points[index + 1]
            
            incoming = current - previous
            outgoing = next_point - current
            incoming_length = abs(incoming.x()) + abs(incoming.y())
            outgoing_length = abs(outgoing.x()) + abs(outgoing.y())
            if (
                (incoming.x() == 0 and outgoing.x() == 0)
                or (incoming.y() == 0 and outgoing.y() == 0)
            ):
                path.lineTo(current)
                continue
            
            radius = min(
                self.ORTHOGONAL_CORNER_RADIUS,
                incoming_length / 2,
                outgoing_length / 2,
            )
            if radius <= 0:
                path.lineTo(current)
                continue
            
            incoming_unit = QPointF(
                0.0 if incoming.x() == 0 else incoming.x() / abs(incoming.x()),
                0.0 if incoming.y() == 0 else incoming.y() / abs(incoming.y()),
            )
            outgoing_unit = QPointF(
                0.0 if outgoing.x() == 0 else outgoing.x() / abs(outgoing.x()),
                0.0 if outgoing.y() == 0 else outgoing.y() / abs(outgoing.y()),
            )
            
            corner_start = current - incoming_unit * radius
            corner_end = current + outgoing_unit * radius
            path.lineTo(corner_start)
            path.quadTo(current, corner_end)
        
        path.lineTo(compact_points[-1])
    
    def update_path(self):
        """Update the connection as a lightly rounded orthogonal path."""
        start = self.source_port.scenePos()
        end = self.target_port.scenePos()
        
        path = QPainterPath()
        
        source_direction = self._port_exit_direction(self.source_port)
        target_direction = self._port_exit_direction(self.target_port)
        start_exit = QPointF(
            start.x() + source_direction * self.ORTHOGONAL_EXIT_LENGTH,
            start.y(),
        )
        end_entry = QPointF(
            end.x() + target_direction * self.ORTHOGONAL_EXIT_LENGTH,
            end.y(),
        )
        
        route_offset = self._parallel_route_offset()
        if end.x() < start.x():
            points = self._reverse_route_points(start, end, route_offset)
        else:
            middle_x = (start_exit.x() + end_entry.x()) / 2 + route_offset
            points = [
                start,
                start_exit,
                QPointF(middle_x, start.y()),
                QPointF(middle_x, end.y()),
                end_entry,
                end,
            ]
        
        self._append_rounded_corner_path(path, points)
        self.setPath(path)
    
    def remove(self):
        """从端口连接列表中移除自己"""
        if self in self.source_port.connections:
            self.source_port.connections.remove(self)
        if self in self.target_port.connections:
            self.target_port.connections.remove(self)


class BaseGraphScene(QGraphicsScene):
    """
    图编辑器场景基类
    
    提供通用的网格背景绘制和连接线绘制逻辑。
    子类需要实现具体的节点/层管理。
    """
    
    GRID_SIZE = 20
    GRID_COLOR = "#252535"
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSceneRect(-5000, -5000, 10000, 10000)
        self.setBackgroundBrush(QBrush(QColor(Styles.COLORS['base'])))
        
        # 连接绘制状态
        self._drawing_connection = False
        self._temp_line: Optional[QGraphicsLineItem] = None
        self._start_port: Optional[BasePortItem] = None
        self._start_item = None  # 起始节点/层项
    
    def drawBackground(self, painter: QPainter, rect: QRectF):
        """绘制网格背景"""
        super().drawBackground(painter, rect)
        
        grid_size = self.GRID_SIZE
        left = int(rect.left()) - (int(rect.left()) % grid_size)
        top = int(rect.top()) - (int(rect.top()) % grid_size)
        
        lines = []
        for x in range(left, int(rect.right()), grid_size):
            lines.append(((x, rect.top()), (x, rect.bottom())))
        for y in range(top, int(rect.bottom()), grid_size):
            lines.append(((rect.left(), y), (rect.right(), y)))
        
        painter.setPen(QPen(QColor(self.GRID_COLOR), 1))
        for line in lines:
            painter.drawLine(QPointF(*line[0]), QPointF(*line[1]))
    
    def start_connection_drawing(self, port: BasePortItem, pos: QPointF):
        """
        开始绘制连接线
        
        Args:
            port: 起始端口
            pos: 鼠标位置
        """
        self._drawing_connection = True
        self._start_port = port
        self._start_item = port.parentItem()
        
        # 创建临时线
        self._temp_line = QGraphicsLineItem()
        self._temp_line.setPen(
            QPen(QColor(Styles.COLORS['yellow']), 2, Qt.PenStyle.DashLine)
        )
        port_pos = port.scenePos()
        self._temp_line.setLine(port_pos.x(), port_pos.y(), pos.x(), pos.y())
        self.addItem(self._temp_line)
    
    def update_connection_drawing(self, pos: QPointF):
        """更新临时连接线位置"""
        if self._drawing_connection and self._temp_line:
            line = self._temp_line.line()
            self._temp_line.setLine(line.x1(), line.y1(), pos.x(), pos.y())
    
    def end_connection_drawing(self):
        """结束连接线绘制"""
        if self._temp_line:
            self.removeItem(self._temp_line)
            self._temp_line = None
        
        start_port = self._start_port
        start_item = self._start_item
        
        self._drawing_connection = False
        self._start_port = None
        self._start_item = None
        
        return start_port, start_item
    
    @property
    def is_drawing_connection(self) -> bool:
        """是否正在绘制连接"""
        return self._drawing_connection


class BaseGraphView(QGraphicsView):
    """
    图编辑器视图基类
    
    提供通用的缩放、平移和拖放功能。
    """
    
    ZOOM_FACTOR = 1.15
    
    def __init__(self, scene: QGraphicsScene, parent=None):
        super().__init__(scene, parent)
        
        # 渲染设置
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        
        # 允许拖放
        self.setAcceptDrops(True)
        
        # 平移状态
        self._is_panning = False
        self._last_pan_pos = QPoint()

    def fit_items_to_origin(self, movable_items: Iterable[QGraphicsItem]):
        """Fit graph content and rebase movable items so the content center is the origin."""
        items = list(movable_items)
        if not items:
            self.resetTransform()
            self.centerOn(0, 0)
            return

        scene = self.scene()
        content_rect = scene.itemsBoundingRect()
        if content_rect.isNull() or content_rect.isEmpty():
            self.resetTransform()
            self.centerOn(0, 0)
            return

        offset = content_rect.center()
        for item in items:
            item.setPos(item.pos() - offset)

        rebased_rect = scene.itemsBoundingRect()
        self.resetTransform()
        self.fitInView(rebased_rect, Qt.AspectRatioMode.KeepAspectRatio)
        self.centerOn(0, 0)
    
    def wheelEvent(self, event: QWheelEvent):
        """鼠标滚轮缩放"""
        factor = self.ZOOM_FACTOR if event.angleDelta().y() > 0 else 1 / self.ZOOM_FACTOR
        self.scale(factor, factor)
    
    def mousePressEvent(self, event: QMouseEvent):
        """鼠标按下 - 中键平移"""
        if event.button() == Qt.MouseButton.MiddleButton:
            self._is_panning = True
            self._last_pan_pos = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        else:
            super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """鼠标移动 - 平移"""
        if self._is_panning:
            delta = event.pos() - self._last_pan_pos
            self._last_pan_pos = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
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
        """Keyboard shortcuts for graph editing."""
        parent = self.parent()

        if event.matches(QKeySequence.StandardKey.Copy):
            if hasattr(parent, '_copy_selected') and parent._copy_selected():
                return

        if event.matches(QKeySequence.StandardKey.Paste):
            if hasattr(parent, '_paste_clipboard') and parent._paste_clipboard():
                return

        if event.key() == Qt.Key.Key_Delete or event.key() == Qt.Key.Key_Backspace:
            if hasattr(parent, '_delete_selected'):
                parent._delete_selected()
            return
        super().keyPressEvent(event)
