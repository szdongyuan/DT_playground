import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLabel, QMenuBar, QPushButton, QWidget

from src.app import AudioTrainingApp


def test_main_window_uses_compact_custom_title_bar():
    app = QApplication.instance() or QApplication([])
    window = AudioTrainingApp()

    try:
        assert window.windowFlags() & Qt.WindowType.FramelessWindowHint

        title_bar = window.findChild(QWidget, "appTitleBar")
        assert title_bar is not None
        assert 42 <= title_bar.height() <= 48

        title_label = window.findChild(QLabel, "appTitleLabel")
        assert title_label is not None
        assert title_label.text() == "AI 声学信号训练平台"

        selected_tab = window.findChild(QPushButton, "titleTab_workflow")
        assert selected_tab is not None
        assert selected_tab.isChecked()
    finally:
        window.close()
        app.processEvents()


def test_maximize_restore_uses_custom_state_when_qt_state_lags(monkeypatch):
    app = QApplication.instance() or QApplication([])
    window = AudioTrainingApp()

    try:
        window._toggle_maximized()
        app.processEvents()

        assert window._custom_is_maximized is True

        monkeypatch.setattr(window, "isMaximized", lambda: False)

        window._toggle_maximized()
        app.processEvents()

        assert window._custom_is_maximized is False
        assert not (window.windowState() & Qt.WindowState.WindowMaximized)
    finally:
        window.close()
        app.processEvents()


def test_settings_action_is_moved_to_title_bar_button(monkeypatch):
    app = QApplication.instance() or QApplication([])
    window = AudioTrainingApp()

    try:
        opened = []

        class FakeSignal:
            def connect(self, callback):
                self.callback = callback

        class FakeSettingsDialog:
            def __init__(self, parent=None):
                self.settings_changed = FakeSignal()

            def exec(self):
                opened.append(True)

        monkeypatch.setattr("src.app.SettingsDialog", FakeSettingsDialog)

        settings_button = window.findChild(QPushButton, "titleSettingsButton")
        assert settings_button is not None
        assert settings_button.text() == "⚙"

        menu_bar = window.findChild(QMenuBar)
        top_level_menu_texts = [action.text() for action in menu_bar.actions()]
        assert all("Edit" not in text and "编辑" not in text for text in top_level_menu_texts)

        settings_button.click()

        assert opened == [True]
    finally:
        window.close()
        app.processEvents()
