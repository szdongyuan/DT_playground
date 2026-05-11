import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QMessageBox, QSplitter, QTabWidget

from src.ui.model_editor.layer_palette import LayerPalette
from src.ui.model_editor.layer_property_panel import LayerPropertyPanel
from src.ui.node_editor import NodePalette, PropertyPanel
from src.ui.styles import Styles
from src.ui.views.model_builder_view import ModelBuilderView
from src.ui.views.workflow_tabs_view import WorkflowTabsView
from src.utils.config import config
from src.workflow.workflow import Workflow


def _app():
    return QApplication.instance() or QApplication([])


def test_canvas_tab_style_is_shared_and_uses_dark_ide_palette():
    style = Styles.canvas_tab_widget()

    assert "QTabBar::tab" in style
    assert f"color: {Styles.COLORS['text']}" in style
    assert f"background: {Styles.COLORS['surface2']}" in style
    assert "background: rgba(49, 50, 68, 0.62)" in style
    assert f"border-bottom: 2px solid {Styles.COLORS['blue']}" in style
    assert f"border-bottom: 1px solid {Styles.COLORS['surface1']}" in style
    assert f"border-bottom: 2px solid {Styles.COLORS['lavender']}" in style


def test_workflow_workspace_keeps_tabs_inside_center_canvas_column():
    _app()
    view = WorkflowTabsView()

    try:
        assert isinstance(view._workspace_splitter, QSplitter)
        assert isinstance(view._node_palette, NodePalette)
        assert isinstance(view._property_panel, PropertyPanel)
        assert isinstance(view._tabs, QTabWidget)

        assert view._workspace_splitter.indexOf(view._node_palette) == 0
        assert view._workspace_splitter.indexOf(view._canvas_column) == 1
        assert view._workspace_splitter.indexOf(view._property_panel) == 2
        assert view._tabs.parentWidget() is view._canvas_column
    finally:
        view.close()
        _app().processEvents()


def test_model_builder_starts_with_center_canvas_tabs_and_can_add_model_tab():
    _app()
    view = ModelBuilderView()

    try:
        assert isinstance(view._workspace_splitter, QSplitter)
        assert isinstance(view._layer_palette, LayerPalette)
        assert isinstance(view._property_panel, LayerPropertyPanel)
        assert isinstance(view._tabs, QTabWidget)

        assert view._workspace_splitter.indexOf(view._layer_palette) == 0
        assert view._workspace_splitter.indexOf(view._canvas_column) == 1
        assert view._workspace_splitter.indexOf(view._property_panel) == 2
        assert view._tabs.count() == 1

        first_graph = view.get_model_graph()
        view._on_new()

        assert view._tabs.count() == 2
        assert view.get_model_graph() is not first_graph
    finally:
        view.close()
        _app().processEvents()


def test_model_builder_persists_file_backed_model_tabs(monkeypatch, tmp_path):
    _app()
    view = ModelBuilderView()
    captured = {}

    def capture_set(key, value):
        captured[key] = value

    first = tmp_path / "first.model.json"
    second = tmp_path / "second.model.json"
    first.write_text("{}", encoding="utf-8")
    second.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(config, "set", capture_set)

    try:
        view._current_tab().file_path = str(first)
        view._on_new()
        view._current_tab().file_path = str(second)
        view.persist_session_state_guarded(force=True)

        assert captured["session.open_model_paths"] == [
            str(first.resolve()),
            str(second.resolve()),
        ]
        assert captured["session.active_model_tab"] == 1
    finally:
        view.close()
        _app().processEvents()


def test_model_builder_save_updates_file_backed_tab_session(monkeypatch, tmp_path):
    _app()
    view = ModelBuilderView()
    captured = {}
    target = tmp_path / "saved.model.json"

    monkeypatch.setattr(config, "set", lambda key, value: captured.__setitem__(key, value))
    monkeypatch.setattr("src.ui.views.model_builder_view.QMessageBox.information", lambda *args, **kwargs: None)
    view.set_session_restore_in_progress(False)

    try:
        assert view._save_to_path(str(target)) is True
        assert captured["session.open_model_paths"] == [str(target.resolve())]
        assert captured["session.active_model_tab"] == 0
    finally:
        view.close()
        _app().processEvents()


def test_model_builder_save_failure_reports_false(monkeypatch, tmp_path):
    _app()
    view = ModelBuilderView()

    def fail_save(_path):
        raise OSError("cannot save")

    monkeypatch.setattr(view.get_model_graph(), "save", fail_save)
    monkeypatch.setattr("src.ui.views.model_builder_view.QMessageBox.critical", lambda *args, **kwargs: None)

    try:
        assert view._save_to_path(str(tmp_path / "broken.model.json")) is False
    finally:
        view.close()
        _app().processEvents()


def test_workflow_replacing_workflow_clears_shared_property_panel():
    _app()
    view = WorkflowTabsView()
    view._property_panel._current_node = object()
    view._property_panel._current_node_id = "stale"

    try:
        view.set_workflow(Workflow("replacement"))

        assert view._property_panel._current_node is None
        assert view._property_panel._current_node_id is None
    finally:
        view.close()
        _app().processEvents()


def test_workflow_parameter_change_refreshes_dirty_tab_title():
    _app()
    view = WorkflowTabsView()

    try:
        assert not view._tabs.tabText(view._tabs.currentIndex()).endswith("*")

        view._on_parameter_changed("missing", "value", 1)

        assert view._tabs.tabText(view._tabs.currentIndex()).endswith("*")
    finally:
        view.close()
        _app().processEvents()


def test_workflow_clear_current_drops_file_backed_session_path(monkeypatch, tmp_path):
    _app()
    view = WorkflowTabsView()
    captured = {}
    workflow_path = tmp_path / "old_workflow.json"
    workflow_path.write_text("{}", encoding="utf-8")
    editor = view._current_editor()

    monkeypatch.setattr(
        "src.ui.views.workflow_tabs_view.QMessageBox.question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Yes,
    )
    monkeypatch.setattr(config, "set", lambda key, value: captured.__setitem__(key, value))

    try:
        view._set_editor_session_file_path(editor, str(workflow_path))
        view._clear_current()
        view.persist_session_state_guarded(force=True)

        assert captured["session.open_workflow_paths"] == []
        assert captured["session.active_workflow_tab"] == -1
    finally:
        view.close()
        _app().processEvents()
