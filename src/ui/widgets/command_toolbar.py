# -*- coding: utf-8 -*-
"""Shared grouped command toolbar widgets."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from src.ui.styles import Styles


class GroupedCommandToolbar(QFrame):
    """Compact dark toolbar with titled command groups."""

    def __init__(self, object_name: str, parent=None):
        super().__init__(parent)
        self.setObjectName(object_name)
        self.setFixedHeight(60)
        self.setStyleSheet(f"""
            QFrame#{object_name} {{
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #111827,
                    stop: 0.55 {Styles.COLORS['mantle']},
                    stop: 1 #151a2c
                );
                border-top: 1px solid rgba(88, 91, 112, 0.38);
                border-bottom: 1px solid rgba(88, 91, 112, 0.72);
            }}
            QLabel[role="groupTitle"] {{
                color: {Styles.COLORS['subtext1']};
                font-size: 11px;
                font-weight: 600;
                padding: 0 1px;
            }}
            QFrame[role="toolbarSeparator"] {{
                background: rgba(88, 91, 112, 0.58);
                min-width: 1px;
                max-width: 1px;
            }}
        """)

        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(16, 5, 16, 5)
        self._layout.setSpacing(16)

    def add_group(self, key: str, title: str, buttons: list[QPushButton]) -> None:
        group = QWidget(self)
        group.setObjectName(f"toolbarGroup_{key}")
        group_layout = QVBoxLayout(group)
        group_layout.setContentsMargins(0, 0, 0, 0)
        group_layout.setSpacing(4)

        title_label = QLabel(title, group)
        title_label.setObjectName(f"toolbarGroupTitle_{key}")
        title_label.setProperty("role", "groupTitle")
        group_layout.addWidget(title_label, 0, Qt.AlignmentFlag.AlignLeft)

        button_row = QWidget(group)
        button_row.setObjectName(f"toolbarGroupButtons_{key}")
        button_layout = QHBoxLayout(button_row)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(6)
        for button in buttons:
            button_layout.addWidget(button)
        group_layout.addWidget(button_row)

        if self._layout.count() > 0:
            self._layout.addWidget(self._separator())
        self._layout.addWidget(group, 0, Qt.AlignmentFlag.AlignVCenter)

    def add_end_stretch(self) -> None:
        self._layout.addStretch(1)

    def _separator(self) -> QFrame:
        separator = QFrame(self)
        separator.setProperty("role", "toolbarSeparator")
        separator.setFixedHeight(38)
        return separator


def command_button(
    text: str,
    object_name: str,
    *,
    variant: str = "normal",
    enabled: bool = True,
    parent=None,
) -> QPushButton:
    """Create a consistently styled toolbar command button."""
    button = QPushButton(text, parent)
    button.setObjectName(object_name)
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    button.setFixedHeight(26)
    button.setMinimumWidth(58)
    button.setEnabled(enabled)

    palettes = {
        "normal": ("#252b3b", "#3a4257", Styles.COLORS["text"], "#596176"),
        "primary": ("#2d6a4f", "#40916c", "#f4fff8", "#52b788"),
        "danger": ("#5a2530", "#7a3442", "#ffe7ec", "#9a4556"),
        "warning": ("#57402f", "#73543c", "#fff3df", "#8a6749"),
    }
    bg, hover, fg, border = palettes.get(variant, palettes["normal"])
    disabled_fg = Styles.COLORS["overlay0"]

    button.setStyleSheet(f"""
        QPushButton#{object_name} {{
            background: {bg};
            border: 1px solid {border};
            border-radius: 5px;
            color: {fg};
            font-size: 12px;
            font-weight: 500;
            padding: 4px 9px;
        }}
        QPushButton#{object_name}:hover {{
            background: {hover};
            border-color: {Styles.COLORS['blue']};
        }}
        QPushButton#{object_name}:pressed {{
            background: {Styles.COLORS['surface1']};
        }}
        QPushButton#{object_name}:disabled {{
            background: #1b2130;
            border-color: #30364a;
            color: {disabled_fg};
        }}
    """)
    return button
