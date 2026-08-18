import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from src.controllers.workflow_controller import WorkflowController
from src.ui.styles import Styles
from src.ui.views.workflow_tabs_view import WorkflowTabsView


def test_workflow_tab_bar_styles_keep_inactive_tab_text_readable():
    app = QApplication.instance() or QApplication([])
    view = WorkflowTabsView()

    try:
        style = view._tabs.styleSheet()

        assert "QTabBar::tab" in style
        assert f"color: {Styles.COLORS['text']}" in style
        assert f"background: {Styles.COLORS['surface2']}" in style
        assert "background: rgba(49, 50, 68, 0.62)" in style
        assert f"border-bottom: 2px solid {Styles.COLORS['blue']}" in style
        assert f"border-bottom: 1px solid {Styles.COLORS['surface1']}" in style
    finally:
        view.close()
        app.processEvents()


def test_validation_failure_after_completed_run_restores_idle_controls():
    app = QApplication.instance() or QApplication([])
    view = WorkflowTabsView()
    view.set_workflow_controller(WorkflowController())

    try:
        tab_index = view.get_current_tab_index()

        # First run starts and completes normally.
        view.set_running_tab_index(tab_index)
        view.set_run_controls_state(run_enabled=False, stop_enabled=True)
        view.clear_running_tab()
        view.set_run_controls_state(run_enabled=True, stop_enabled=False)

        # A second run marks the tab before pre-run validation rejects it.
        view.set_running_tab_index(tab_index)
        view.clear_running_tab()

        assert view._toolbar._run_btn.isEnabled()
        assert not view._toolbar._stop_btn.isEnabled()

        # Clicking stop while no execution exists must remain an idle no-op.
        view._on_stop_workflow()

        assert view._toolbar._run_btn.isEnabled()
        assert not view._toolbar._stop_btn.isEnabled()
        assert view._toolbar._stop_btn.text().endswith("停止")
    finally:
        view.close()
        app.processEvents()
