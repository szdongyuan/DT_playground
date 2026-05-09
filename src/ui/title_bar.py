# -*- coding: utf-8 -*-
"""Custom frameless application title bar."""

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from src.ui.styles import Styles


class AppSignalIcon(QWidget):
    """Small rounded icon with a signal waveform motif."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(24, 24)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect().adjusted(1, 1, -1, -1)
        gradient = QLinearGradient(rect.topLeft(), rect.bottomRight())
        gradient.setColorAt(0.0, QColor("#6d5dfc"))
        gradient.setColorAt(0.55, QColor("#3f8cff"))
        gradient.setColorAt(1.0, QColor("#cba6f7"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(gradient)
        painter.drawRoundedRect(rect, 6, 6)

        pen = QPen(QColor("#eef4ff"), 1.4)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)

        center_y = rect.center().y()
        x_values = [6, 9, 12, 15, 18]
        heights = [5, 10, 14, 9, 6]
        for x, height in zip(x_values, heights):
            painter.drawLine(x, center_y - height // 2, x, center_y + height // 2)

        painter.end()


class TitleNavButton(QPushButton):
    """Top-level navigation tab used in the custom title bar."""

    def __init__(self, text: str, object_name: str, parent=None):
        super().__init__(text, parent)
        self.setObjectName(object_name)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(38)
        self.setMinimumWidth(78)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                border: none;
                border-bottom: 2px solid transparent;
                color: {Styles.COLORS['subtext1']};
                font-size: 14px;
                font-weight: 500;
                padding: 8px 18px 7px 18px;
            }}
            QPushButton:hover {{
                background: rgba(69, 71, 90, 0.45);
                color: {Styles.COLORS['text']};
            }}
            QPushButton:checked {{
                background: rgba(49, 50, 68, 0.88);
                border-bottom: 2px solid {Styles.COLORS['blue']};
                color: #f4f7ff;
                font-weight: 600;
            }}
        """)


class WindowControlButton(QPushButton):
    """Small caption button for a frameless window."""

    def __init__(self, text: str, object_name: str, parent=None, *, danger: bool = False):
        super().__init__(text, parent)
        self.setObjectName(object_name)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(36, 30)
        hover = "rgba(243, 139, 168, 0.85)" if danger else "rgba(88, 91, 112, 0.55)"
        active = Styles.COLORS["red"] if danger else "rgba(108, 112, 134, 0.7)"
        self.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                border: none;
                color: {Styles.COLORS['subtext1']};
                font-size: 13px;
                font-weight: 400;
            }}
            QPushButton:hover {{
                background: {hover};
                color: #ffffff;
            }}
            QPushButton:pressed {{
                background: {active};
            }}
        """)


class AppTitleBar(QFrame):
    """Compact custom title bar with app branding, navigation and window controls."""

    view_changed = Signal(int)
    settings_requested = Signal()
    minimize_requested = Signal()
    maximize_restore_requested = Signal()
    close_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("appTitleBar")
        self.setFixedHeight(46)
        self._drag_position: QPoint | None = None
        self._setup_ui()

    def set_current_view(self, index: int) -> None:
        button = self._nav_group.button(index)
        if button is not None:
            button.setChecked(True)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            window = self.window()
            self._drag_position = (
                event.globalPosition().toPoint() - window.frameGeometry().topLeft()
            )
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if (
            event.buttons() & Qt.MouseButton.LeftButton
            and self._drag_position is not None
            and not self.window().isMaximized()
        ):
            self.window().move(event.globalPosition().toPoint() - self._drag_position)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_position = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.maximize_restore_requested.emit()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def _setup_ui(self) -> None:
        self.setStyleSheet(f"""
            QFrame#appTitleBar {{
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #0b1020,
                    stop: 0.48 {Styles.COLORS['mantle']},
                    stop: 1 #21172f
                );
                border-bottom: 1px solid rgba(88, 91, 112, 0.55);
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 8, 0)
        layout.setSpacing(10)

        icon = AppSignalIcon(self)
        layout.addWidget(icon, 0, Qt.AlignmentFlag.AlignVCenter)

        title = QLabel("AI 声学信号训练平台", self)
        title.setObjectName("appTitleLabel")
        title.setStyleSheet("""
            QLabel#appTitleLabel {
                color: #f4f7ff;
                font-size: 16px;
                font-weight: 600;
                letter-spacing: 0.2px;
            }
        """)
        layout.addWidget(title, 0, Qt.AlignmentFlag.AlignVCenter)

        layout.addSpacing(54)

        self._nav_group = QButtonGroup(self)
        self._nav_group.setExclusive(True)
        tabs = [
            ("工作流", "titleTab_workflow"),
            ("模型", "titleTab_model"),
            ("预览", "titleTab_preview"),
            ("训练", "titleTab_training"),
        ]
        for index, (text, object_name) in enumerate(tabs):
            button = TitleNavButton(text, object_name, self)
            self._nav_group.addButton(button, index)
            layout.addWidget(button, 0, Qt.AlignmentFlag.AlignVCenter)

        self._nav_group.button(0).setChecked(True)
        self._nav_group.idClicked.connect(self.view_changed.emit)

        layout.addStretch(1)

        settings_btn = WindowControlButton("⚙", "titleSettingsButton", self)
        minimize_btn = WindowControlButton("–", "windowMinimizeButton", self)
        maximize_btn = WindowControlButton("□", "windowMaximizeButton", self)
        close_btn = WindowControlButton("×", "windowCloseButton", self, danger=True)

        settings_btn.clicked.connect(self.settings_requested)
        minimize_btn.clicked.connect(self.minimize_requested)
        maximize_btn.clicked.connect(self.maximize_restore_requested)
        close_btn.clicked.connect(self.close_requested)

        layout.addWidget(settings_btn, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(minimize_btn, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(maximize_btn, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(close_btn, 0, Qt.AlignmentFlag.AlignVCenter)
