"""
Startup splash screen helpers.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPixmap
from PyQt6.QtWidgets import QApplication, QLabel, QProgressBar, QSplashScreen


def get_app_root_path() -> Path:
    """Return the portable app root path for dev and frozen runs."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def get_splash_image_path() -> Path:
    """Return the portable splash image path for dev and frozen runs."""
    return get_app_root_path() / "assets" / "splash_screen.png"


def get_app_icon_path() -> Path:
    """Return the portable application icon path for dev and frozen runs."""
    return get_app_root_path() / "assets" / "DTPG_logo.ico"


class StartupSplash(QSplashScreen):
    """Splash screen with an overlaid progress bar and status text."""

    def __init__(self):
        super().__init__(self._load_pixmap())
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.setObjectName("startupSplash")
        self._message_label = QLabel(self)
        self._message_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self._message_label.setStyleSheet(
            "color: white; background: transparent; font-size: 16px; font-weight: 600;"
        )

        self._progress_bar = QProgressBar(self)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setStyleSheet(
            """
            QProgressBar {
                color: white;
                background-color: rgba(8, 17, 34, 180);
                border: 1px solid rgba(255, 255, 255, 80);
                border-radius: 8px;
                text-align: center;
                padding: 2px;
            }
            QProgressBar::chunk {
                background-color: rgba(105, 190, 255, 220);
                border-radius: 6px;
            }
            """
        )

        self._hint_label = QLabel("Replace assets/splash_screen.png to customize", self)
        self._hint_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self._hint_label.setStyleSheet(
            "color: rgba(255, 255, 255, 190); background: transparent; font-size: 11px;"
        )
        self._layout_overlay()

    @staticmethod
    def _load_pixmap() -> QPixmap:
        """Load the configured splash image or return a generated fallback."""
        image_path = get_splash_image_path()
        if image_path.exists():
            pixmap = QPixmap(os.fspath(image_path))
            if not pixmap.isNull():
                return pixmap
        return StartupSplash._build_fallback_pixmap(image_path)

    @staticmethod
    def _build_fallback_pixmap(image_path) -> QPixmap:
        """Build a basic branded fallback image when no PNG is available."""
        pixmap = QPixmap(1024, 720)
        pixmap.fill(QColor("#07111f"))

        painter = QPainter(pixmap)
        gradient = QLinearGradient(0, 0, pixmap.width(), pixmap.height())
        gradient.setColorAt(0.0, QColor("#07111f"))
        gradient.setColorAt(0.55, QColor("#0c2f63"))
        gradient.setColorAt(1.0, QColor("#1590e5"))
        painter.fillRect(pixmap.rect(), gradient)

        painter.setPen(QColor(255, 255, 255, 235))
        title_font = QFont("Microsoft YaHei", 26)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.drawText(
            pixmap.rect().adjusted(56, 56, -56, -180),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom,
            "AI Acoustic Training Platform",
        )

        subtitle_font = QFont("Microsoft YaHei", 12)
        painter.setFont(subtitle_font)
        painter.setPen(QColor(255, 255, 255, 190))
        painter.drawText(
            pixmap.rect().adjusted(56, 56, -56, -145),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom,
            f"Missing {image_path}, using fallback splash.",
        )
        painter.end()
        return pixmap

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._layout_overlay()

    def _layout_overlay(self):
        """Place the overlay widgets near the bottom of the splash image."""
        width = self.pixmap().width()
        height = self.pixmap().height()
        left = 48
        bar_width = max(320, width - 96)

        self._message_label.setGeometry(left, height - 110, bar_width, 26)
        self._progress_bar.setGeometry(left, height - 78, bar_width, 24)
        self._hint_label.setGeometry(left, height - 48, bar_width, 18)

    def set_progress(self, value: int, step_text: str | None = None):
        """Update progress state and process pending paint events."""
        bounded_value = max(0, min(100, int(value)))
        self._progress_bar.setValue(bounded_value)
        if step_text:
            self._message_label.setText(step_text)
        QApplication.processEvents()

