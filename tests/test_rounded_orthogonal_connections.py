import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QPainterPath
from PySide6.QtWidgets import QApplication, QGraphicsRectItem, QGraphicsScene

from src.ui.graph_editor.base_items import BaseConnectionItem, BasePortItem
from src.ui.styles import Styles


def _path_elements(path):
    return [path.elementAt(index) for index in range(path.elementCount())]


def _line_points(path):
    return [
        (path.elementAt(index).x, path.elementAt(index).y)
        for index in range(path.elementCount())
        if path.elementAt(index).type == QPainterPath.ElementType.LineToElement
    ]


def test_connection_path_uses_rounded_orthogonal_segments():
    app = QApplication.instance() or QApplication([])
    scene = QGraphicsScene()

    source = BasePortItem(is_input=False, color=Styles.COLORS["green"])
    target = BasePortItem(is_input=True, color=Styles.COLORS["blue"])
    scene.addItem(source)
    scene.addItem(target)
    source.setPos(0, 0)
    target.setPos(160, 80)

    connection = BaseConnectionItem(source, target)
    scene.addItem(connection)

    try:
        elements = _path_elements(connection.path())
        line_elements = [
            element for element in elements
            if element.type == QPainterPath.ElementType.LineToElement
        ]
        curve_elements = [
            element for element in elements
            if element.type == QPainterPath.ElementType.CurveToElement
        ]

        assert elements[0].type == QPainterPath.ElementType.MoveToElement
        assert elements[1].type == QPainterPath.ElementType.LineToElement
        assert elements[1].y == elements[0].y
        assert len(line_elements) >= 3
        assert len(curve_elements) >= 1
        assert elements[-1].type == QPainterPath.ElementType.LineToElement
        assert elements[-1].x == target.scenePos().x()
        assert elements[-1].y == target.scenePos().y()
    finally:
        scene.removeItem(connection)
        connection.remove()
        app.processEvents()


def test_reverse_connection_routes_around_outside_of_node_bodies():
    app = QApplication.instance() or QApplication([])
    scene = QGraphicsScene()

    target_node = QGraphicsRectItem(0, 0, 100, 60)
    source_node = QGraphicsRectItem(0, 0, 100, 60)
    scene.addItem(target_node)
    scene.addItem(source_node)
    target_node.setPos(0, 0)
    source_node.setPos(180, 20)

    source = BasePortItem(
        is_input=False,
        color=Styles.COLORS["green"],
        parent=source_node,
    )
    target = BasePortItem(
        is_input=True,
        color=Styles.COLORS["blue"],
        parent=target_node,
    )
    source.setPos(100, 30)
    target.setPos(0, 30)

    connection = BaseConnectionItem(source, target)
    scene.addItem(connection)

    try:
        source_pos = source.scenePos()
        target_pos = target.scenePos()
        source_bounds = source_node.sceneBoundingRect()
        target_bounds = target_node.sceneBoundingRect()
        outer_bottom = max(source_bounds.bottom(), target_bounds.bottom())
        outer_top = min(source_bounds.top(), target_bounds.top())
        line_points = _line_points(connection.path())

        assert target_pos.x() < source_pos.x()
        assert line_points[0][0] > source_bounds.right()
        assert line_points[1][0] > source_bounds.right()
        assert line_points[1][0] >= line_points[0][0]
        assert any(
            y > outer_bottom + 20 or y < outer_top - 20
            for _, y in line_points
        )
        assert all(
            not (
                target_bounds.right() < x < source_bounds.left()
                and outer_top < y < outer_bottom
            )
            for x, y in line_points
        )
        assert line_points[-1] == (target_pos.x(), target_pos.y())
    finally:
        scene.removeItem(connection)
        connection.remove()
        app.processEvents()


def test_staggered_reverse_connection_uses_inter_node_corridor():
    app = QApplication.instance() or QApplication([])
    scene = QGraphicsScene()

    source_node = QGraphicsRectItem(0, 0, 180, 60)
    target_node = QGraphicsRectItem(0, 0, 180, 60)
    scene.addItem(source_node)
    scene.addItem(target_node)
    source_node.setPos(120, 0)
    target_node.setPos(100, 100)

    source = BasePortItem(
        is_input=False,
        color=Styles.COLORS["green"],
        parent=source_node,
    )
    target = BasePortItem(
        is_input=True,
        color=Styles.COLORS["blue"],
        parent=target_node,
    )
    source.setPos(180, 30)
    target.setPos(0, 30)

    connection = BaseConnectionItem(source, target)
    scene.addItem(connection)

    try:
        source_pos = source.scenePos()
        target_pos = target.scenePos()
        source_bounds = source_node.sceneBoundingRect()
        target_bounds = target_node.sceneBoundingRect()
        corridor_top = source_bounds.bottom()
        corridor_bottom = target_bounds.top()
        line_points = _line_points(connection.path())
        corridor_points = [
            (x, y) for x, y in line_points
            if corridor_top < y < corridor_bottom
        ]

        assert target_pos.x() < source_pos.x()
        assert corridor_top < corridor_bottom
        assert corridor_points
        assert not any(y > target_bounds.bottom() for _, y in line_points)
        assert line_points[-1] == (target_pos.x(), target_pos.y())
    finally:
        scene.removeItem(connection)
        connection.remove()
        app.processEvents()
