import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from src.ui.styles import Styles
from src.ui.views.workflow_tabs_view import WorkflowTabsView


def test_workflow_tab_bar_styles_keep_inactive_tab_text_readable():
    app = QApplication.instance() or QApplication([])
    view = WorkflowTabsView()

    try:
        style = view._tabs.styleSheet()

        assert "QTabBar::tab" in style
        assert f"color: {Styles.COLORS['text']}" in style
        assert f"background: {Styles.COLORS['surface0']}" in style
        assert f"background: {Styles.COLORS['surface2']}" in style
    finally:
        view.close()
        app.processEvents()
