"""
Main Application Class
"""

from collections.abc import Callable

from PySide6.QtCore import QRect, Qt, QSize
from PySide6.QtGui import QAction, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog, QMainWindow, QMenu, QMenuBar,
    QMessageBox, QToolBar, QVBoxLayout, QWidget
)

from src.ui.dialogs.about_dialog import AboutDialog
from src.ui.dialogs.export_dialog import ExportModelDialog
from src.ui.dialogs.settings_dialog import SettingsDialog
from src.ui.i18n import tr_
from src.ui.main_window import MainWindow
from src.ui.startup_splash import get_app_icon_path
from src.ui.title_bar import AppTitleBar
from src.utils.restart_manager import RestartManager
from src.utils.config import config


RECENT_WORKFLOWS_MENU_TEXT = "Recent workflows"
NO_RECENT_WORKFLOWS_TEXT = "(No recent workflows)"
CLEAR_RECENT_WORKFLOWS_TEXT = "Clear recent workflows"


def is_recent_workflow_file(file_path: str) -> bool:
    normalized_path = str(file_path).lower()
    return normalized_path.endswith(".json") and not normalized_path.endswith(".model.json")


def filter_recent_workflow_files(recent_files: list[str]) -> list[str]:
    return [file_path for file_path in recent_files if is_recent_workflow_file(file_path)]


def filter_non_workflow_recent_files(recent_files: list[str]) -> list[str]:
    return [file_path for file_path in recent_files if not is_recent_workflow_file(file_path)]


class AudioTrainingApp(QMainWindow):
    """AI声学信号训练平台主应用程序"""
    
    def __init__(self, startup_progress: Callable[[int, str | None], None] | None = None):
        super().__init__()
        self.current_model = None
        self._restart_manager: RestartManager | None = None
        self._restart_in_progress = False
        self._startup_progress = startup_progress
        self._custom_is_maximized = False
        self._normal_geometry = QRect()
        self._init_ui()
        self._report_startup_progress(84, "Main window created...")
        self._init_menubar()
        self._init_toolbar()
        self._init_connections()
        self._report_startup_progress(90, "Restoring previous session...")
        self._load_settings()
        self._report_startup_progress(96, "Finalizing startup...")

    def _report_startup_progress(self, value: int, step_text: str | None = None):
        """Forward startup milestones to the splash screen when available."""
        if self._startup_progress is None:
            return
        self._startup_progress(value, step_text)
    
    def _init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle(tr_("AI Acoustic Signal Training Platform"))
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        self.setMinimumSize(1280, 800)
        icon_path = get_app_icon_path()
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        
        root_widget = QWidget(self)
        self._root_layout = QVBoxLayout(root_widget)
        self._root_layout.setContentsMargins(0, 0, 0, 0)
        self._root_layout.setSpacing(0)

        self.title_bar = AppTitleBar(self)
        self.title_bar.view_changed.connect(self._switch_view)
        self.title_bar.settings_requested.connect(self._show_settings)
        self.title_bar.minimize_requested.connect(self.showMinimized)
        self.title_bar.maximize_restore_requested.connect(self._toggle_maximized)
        self.title_bar.close_requested.connect(self.close)
        self._root_layout.addWidget(self.title_bar)

        self.main_window = MainWindow(self, startup_progress=self._startup_progress)
        self.main_window.view_changed.connect(self.title_bar.set_current_view)
        self._root_layout.addWidget(self.main_window)
        self.setCentralWidget(root_widget)
        
        # 居中显示
        self._center_window()
    
    def _init_menubar(self):
        """初始化菜单栏"""
        menubar = QMenuBar(self)
        self._root_layout.insertWidget(1, menubar)
        menubar.setStyleSheet("""
            QMenuBar {
                background-color: #181825;
                color: #cdd6f4;
                padding: 4px;
                border-bottom: 1px solid #313244;
            }
            QMenuBar::item {
                padding: 6px 12px;
                border-radius: 4px;
            }
            QMenuBar::item:selected {
                background-color: #313244;
            }
            QMenu {
                background-color: #1e1e2e;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 4px;
                padding: 4px;
            }
            QMenu::item {
                padding: 8px 24px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background-color: #313244;
            }
            QMenu::separator {
                height: 1px;
                background-color: #45475a;
                margin: 4px 8px;
            }
        """)
        
        # === 文件菜单 ===
        file_menu = menubar.addMenu(tr_("File(&F)"))
        
        # 加载模型
        self.action_load_model = QAction(tr_("Load model..."), self)
        self.action_load_model.setShortcut(QKeySequence("Ctrl+L"))
        self.action_load_model.triggered.connect(self._load_model)
        file_menu.addAction(self.action_load_model)
        
        # 导出模型
        self.action_export_model = QAction(tr_("Export model..."), self)
        self.action_export_model.setShortcut(QKeySequence("Ctrl+E"))
        self.action_export_model.triggered.connect(self._export_model)
        file_menu.addAction(self.action_export_model)
        
        file_menu.addSeparator()
        
        # Recent workflows
        self.recent_menu = file_menu.addMenu(tr_(RECENT_WORKFLOWS_MENU_TEXT))
        self._update_recent_menu()
        
        file_menu.addSeparator()
        
        # 重启应用
        self.action_restart = QAction(tr_("Restart(&R)"), self)
        self.action_restart.setShortcut(QKeySequence("Ctrl+Shift+R"))
        self.action_restart.triggered.connect(self._request_restart)
        file_menu.addAction(self.action_restart)

        file_menu.addSeparator()

        # 退出
        self.action_exit = QAction(tr_("Exit(&X)"), self)
        self.action_exit.setShortcut(QKeySequence("Alt+F4"))
        self.action_exit.triggered.connect(self.close)
        file_menu.addAction(self.action_exit)
        
        self.action_settings = QAction(tr_("Settings..."), self)
        self.action_settings.setShortcut(QKeySequence("Ctrl+,"))
        self.action_settings.triggered.connect(self._show_settings)
        self.addAction(self.action_settings)
        
        # === 视图菜单 ===
        view_menu = menubar.addMenu(tr_("View(&V)"))
        
        self.action_view_workflow = QAction(tr_("Workflow view"), self)
        self.action_view_workflow.setShortcut(QKeySequence("Ctrl+1"))
        self.action_view_workflow.triggered.connect(lambda: self._switch_view(0))
        view_menu.addAction(self.action_view_workflow)
        
        self.action_view_model = QAction(tr_("Model view"), self)
        self.action_view_model.setShortcut(QKeySequence("Ctrl+2"))
        self.action_view_model.triggered.connect(lambda: self._switch_view(1))
        view_menu.addAction(self.action_view_model)
        
        self.action_view_preview = QAction(tr_("Preview view"), self)
        self.action_view_preview.setShortcut(QKeySequence("Ctrl+3"))
        self.action_view_preview.triggered.connect(lambda: self._switch_view(2))
        view_menu.addAction(self.action_view_preview)
        
        self.action_view_training = QAction(tr_("Training view"), self)
        self.action_view_training.setShortcut(QKeySequence("Ctrl+4"))
        self.action_view_training.triggered.connect(lambda: self._switch_view(3))
        view_menu.addAction(self.action_view_training)
        
        # === 帮助菜单 ===
        help_menu = menubar.addMenu(tr_("Help(&H)"))
        
        self.action_about = QAction(tr_("About..."), self)
        self.action_about.triggered.connect(self._show_about)
        help_menu.addAction(self.action_about)
    
    def _init_toolbar(self):
        """初始化工具栏（已移除，功能整合到工作流视图工具栏）"""
        pass
    
    def _init_connections(self):
        """初始化信号连接"""
        # 主窗口信号
        self.main_window.training_started.connect(self._on_training_started)
        self.main_window.training_stopped.connect(self._on_training_stopped)
    
    def _load_settings(self):
        """加载设置"""
        # 启动恢复：上次工作流 + 模型编辑器状态
        import os
        from src.model_builder.model_graph import ModelGraph
        self._report_startup_progress(91, "Restoring workflow session...")

        # Prevent the default empty tab from overwriting the previous session state
        # before we restore workflow tabs from config.
        try:
            self.main_window._workflow_view.set_session_restore_in_progress(True)
        except Exception:
            pass

        # ===== 恢复工作流 =====
        try:
            open_paths = config.get("session.open_workflow_paths", [])
            active_idx = config.get("session.active_workflow_tab", -1)

            restored_paths: list[str] = []
            if isinstance(open_paths, list):
                for p in open_paths:
                    if isinstance(p, str) and p and os.path.exists(p):
                        restored_paths.append(p)

            if restored_paths:
                try:
                    # Remove the default empty tab before restoring.
                    self.main_window._workflow_view.clear_all_tabs(ensure_one_tab=False)
                except Exception:
                    pass

                for p in restored_paths:
                    try:
                        self.main_window._workflow_view.open_workflow_file(p)
                    except Exception:
                        pass

                # Restore active tab (index into restored file-backed list).
                try:
                    count = int(self.main_window._workflow_view._tabs.count())
                except Exception:
                    count = 0
                if count > 0:
                    try:
                        idx = int(active_idx)
                    except Exception:
                        idx = -1
                    if idx < 0 or idx >= count:
                        idx = 0
                    try:
                        self.main_window._workflow_view._tabs.setCurrentIndex(idx)
                    except Exception:
                        pass

                wf = self.main_window._workflow_view.get_workflow()
                if wf is not None:
                    try:
                        self.main_window._workflow = wf
                    except Exception:
                        pass
                    try:
                        self.main_window._workflow_controller.set_workflow(wf)
                    except Exception:
                        pass
                    try:
                        self.main_window._update_workflow_status()
                    except Exception:
                        pass
            else:
                # Backward compatibility: restore last_workflow_path (single file)
                last_workflow_path = config.get('session.last_workflow_path')
                if last_workflow_path and os.path.exists(last_workflow_path):
                    try:
                        self.main_window._workflow_view.open_workflow_file(last_workflow_path)
                        wf = self.main_window._workflow_view.get_workflow()
                        if wf is not None:
                            try:
                                self.main_window._workflow = wf
                            except Exception:
                                pass
                            try:
                                self.main_window._workflow_controller.set_workflow(wf)
                            except Exception:
                                pass
                            try:
                                self.main_window._update_workflow_status()
                            except Exception:
                                pass
                    except Exception:
                        pass
        except Exception:
            # 启动恢复失败不应阻断应用启动
            pass
        finally:
            # Unfreeze session persistence and persist the restored final state once.
            try:
                self.main_window._workflow_view.set_session_restore_in_progress(False)
                self.main_window._workflow_view.persist_session_state_guarded()
            except Exception:
                pass

        # ===== 恢复模型编辑器 =====
        try:
            self._report_startup_progress(94, "Restoring model editor state...")
            model_graph = None
            model_file_path = None

            last_model_path = config.get('session.last_model_path')
            if last_model_path and os.path.exists(last_model_path):
                try:
                    model_graph = ModelGraph.load(last_model_path)
                    model_file_path = last_model_path
                except Exception:
                    model_graph = None

            if model_graph is None:
                snapshot = config.get('session.last_model_graph_snapshot')
                if isinstance(snapshot, dict):
                    try:
                        model_graph = ModelGraph.from_dict(snapshot)
                        model_file_path = None
                    except Exception:
                        model_graph = None

            if model_graph is not None:
                try:
                    self.main_window._model_builder_view.set_model_graph(
                        model_graph,
                        file_path=model_file_path
                    )
                except Exception:
                    # 不阻断启动；最差情况保持默认模型
                    pass

            if model_file_path:
                try:
                    config.add_recent_file(model_file_path)
                except Exception:
                    pass
        except Exception:
            pass
        self._report_startup_progress(95, "Startup state restored...")
    
    def _center_window(self):
        """将窗口居中显示"""
        screen = self.screen().availableGeometry()
        window_size = self.geometry()
        x = (screen.width() - window_size.width()) // 2
        y = (screen.height() - window_size.height()) // 2
        self.move(x, y)
    
    # === 文件操作 ===
    
    def _load_model(self):
        """加载模型"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            tr_("Load model"),
            "",
            tr_("Keras models (*.h5 *.keras);;Model definition (*.model.json);;SavedModel (*);;All files (*)")
        )
        if file_path:
            try:
                from tensorflow import keras
                self.current_model = keras.models.load_model(file_path)
                self.main_window.set_current_model(self.current_model)
                QMessageBox.information(
                    self,
                    tr_("Success"),
                    tr_("Model loaded: {path}").format(path=file_path),
                )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    tr_("Error"),
                    tr_("Failed to load model: {error}").format(error=str(e)),
                )
    
    def _export_model(self):
        """导出模型"""
        dialog = ExportModelDialog(self, self.current_model)
        dialog.exec()
    
    def _update_recent_menu(self):
        """Update the recent workflows menu."""
        self.recent_menu.clear()
        recent_workflows = filter_recent_workflow_files(config.get_recent_files())
        
        if not recent_workflows:
            action = QAction(tr_(NO_RECENT_WORKFLOWS_TEXT), self)
            action.setEnabled(False)
            self.recent_menu.addAction(action)
            return
        
        for file_path in recent_workflows:
            action = QAction(file_path, self)
            action.triggered.connect(lambda checked, p=file_path: self._open_recent_workflow(p))
            self.recent_menu.addAction(action)
        
        self.recent_menu.addSeparator()
        clear_action = QAction(tr_(CLEAR_RECENT_WORKFLOWS_TEXT), self)
        clear_action.triggered.connect(self._clear_recent_workflows)
        self.recent_menu.addAction(clear_action)
    
    def _open_recent_workflow(self, file_path: str):
        """Open a recent workflow file."""
        import os
        if os.path.exists(file_path):
            try:
                self.main_window._workflow_view.open_workflow_file(file_path)
                wf = self.main_window._workflow_view.get_workflow()
                if wf is not None:
                    try:
                        self.main_window._workflow = wf
                    except Exception:
                        pass
                    try:
                        self.main_window._workflow_controller.set_workflow(wf)
                    except Exception:
                        pass
                    try:
                        self.main_window._update_workflow_status()
                    except Exception:
                        pass
                    try:
                        config.set('session.last_workflow_path', file_path)
                    except Exception:
                        pass
                self._switch_view(0)
            except Exception:
                pass
        else:
            QMessageBox.warning(
                self,
                tr_("Warning"),
                tr_("File does not exist: {path}").format(path=file_path),
            )

    def _open_recent_file(self, file_path: str):
        """Keep the old internal method name as a workflow-only wrapper."""
        self._open_recent_workflow(file_path)
    
    def _clear_recent_workflows(self):
        """Clear workflow entries while keeping other recent-file history."""
        config.set('recent.files', filter_non_workflow_recent_files(config.get_recent_files()))
        self._update_recent_menu()
    
    # === 视图操作 ===
    
    def _switch_view(self, index: int):
        """切换视图"""
        self.main_window.switch_view(index)
        self.title_bar.set_current_view(index)

    def _toggle_maximized(self):
        """Toggle maximized state for the frameless window."""
        is_maximized = (
            self._custom_is_maximized
            or self.isMaximized()
            or bool(self.windowState() & Qt.WindowState.WindowMaximized)
        )
        if is_maximized:
            self.showNormal()
            if self._normal_geometry.isValid():
                self.setGeometry(self._normal_geometry)
            self._custom_is_maximized = False
        else:
            if not self.isMaximized():
                self._normal_geometry = self.geometry()
            self.showMaximized()
            self._custom_is_maximized = True
    
    # === 工作流操作 ===
    
    def _run_workflow(self):
        """运行工作流"""
        self.main_window._on_run_workflow()
    
    def _on_training_started(self):
        """训练开始回调"""
        pass
    
    def _on_training_stopped(self):
        """训练停止回调"""
        pass
    
    # === 设置和帮助 ===
    
    def _show_settings(self):
        """显示设置对话框"""
        dialog = SettingsDialog(self)
        dialog.settings_changed.connect(self._apply_settings)
        dialog.exec()
    
    def _apply_settings(self, settings: dict):
        """应用设置"""
        old_lang = config.get("ui.language", "zh_CN")
        new_lang = settings.get("ui.language", old_lang)

        # 更新设置
        for key, value in settings.items():
            config.set(key, value, save_immediately=False)
        config.save()

        # If UI language changed, prompt restart now/later.
        if str(new_lang) != str(old_lang):
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Question)
            msg.setWindowTitle(tr_("Restart required"))
            msg.setText(tr_("Language change will take effect after restart."))
            btn_now = msg.addButton(tr_("Restart now"), QMessageBox.ButtonRole.AcceptRole)
            btn_later = msg.addButton(tr_("Restart later"), QMessageBox.ButtonRole.RejectRole)
            try:
                msg.setDefaultButton(btn_later)
            except Exception:
                pass
            msg.exec()

            if msg.clickedButton() == btn_now:
                self._request_restart()
            return
        
        QMessageBox.information(
            self,
            tr_("Info"),
            tr_("Settings saved. Some changes require a restart to take effect."),
        )
    
    def _show_about(self):
        """显示关于对话框"""
        dialog = AboutDialog(self)
        dialog.exec()

    def _request_restart(self):
        """请求重启应用（graceful restart）"""
        if self._restart_in_progress:
            return

        # busy 判定：工作流或训练任一在运行都视为 busy
        workflow_running = False
        training_running = False
        try:
            workflow_running = bool(self.main_window._workflow_controller.is_running())
        except Exception:
            workflow_running = False
        try:
            training_running = bool(self.main_window._training_controller.is_training)
        except Exception:
            training_running = False

        if workflow_running or training_running:
            resp = QMessageBox.question(
                self,
                tr_("Confirm restart"),
                tr_(
                    "A workflow/training is currently running.\n"
                    "Restart will stop the current run and start a new instance.\n\n"
                    "Continue?"
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if resp != QMessageBox.StandardButton.Yes:
                return

        self._restart_in_progress = True
        try:
            self.action_restart.setEnabled(False)
        except Exception:
            pass

        self._restart_manager = RestartManager(parent=self, app_window=self)
        self._restart_manager.restart(
            stop_timeout_s=10.0,
            ready_timeout_s=10.0,
            on_finished=self._on_restart_flow_finished,
        )

    def _on_restart_flow_finished(self, success: bool, message: str):
        """重启流程结束回调（失败时恢复 UI）"""
        self._restart_in_progress = False
        if not success:
            try:
                self.action_restart.setEnabled(True)
            except Exception:
                pass
            QMessageBox.warning(
                self,
                tr_("Restart failed"),
                message + "\n\n" + tr_("Please check the latest log under the logs directory."),
            )
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        # Prompt on unsaved workflows (Save / Don't Save / Cancel)
        try:
            if not self.main_window._workflow_view.confirm_save_dirty_on_close():
                event.ignore()
                return
        except Exception:
            # Never block close if prompt flow fails.
            pass

        # Persist session state (open tabs + active index)
        try:
            self.main_window._workflow_view.persist_session_state_guarded(force=True)
        except Exception:
            pass

        config.save()
        event.accept()
