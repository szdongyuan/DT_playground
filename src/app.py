"""
Main Application Class
"""

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QAction, QIcon, QKeySequence
from PyQt6.QtWidgets import (
    QFileDialog, QMainWindow, QMenu, QMenuBar,
    QMessageBox, QToolBar, QVBoxLayout, QWidget
)

from src.ui.dialogs.about_dialog import AboutDialog
from src.ui.dialogs.export_dialog import ExportModelDialog
from src.ui.dialogs.settings_dialog import SettingsDialog
from src.ui.main_window import MainWindow
from src.utils.restart_manager import RestartManager
from src.utils.config import config


class AudioTrainingApp(QMainWindow):
    """AI声学信号训练平台主应用程序"""
    
    def __init__(self):
        super().__init__()
        self.current_model = None
        self._restart_manager: RestartManager | None = None
        self._restart_in_progress = False
        self._init_ui()
        self._init_menubar()
        self._init_toolbar()
        self._init_connections()
        self._load_settings()
    
    def _init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("AI声学信号训练平台")
        self.setMinimumSize(1280, 800)
        
        # 设置主窗口内容
        self.main_window = MainWindow(self)
        self.setCentralWidget(self.main_window)
        
        # 居中显示
        self._center_window()
    
    def _init_menubar(self):
        """初始化菜单栏"""
        menubar = self.menuBar()
        menubar.setStyleSheet("""
            QMenuBar {
                background-color: #181825;
                color: #cdd6f4;
                padding: 4px;
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
        file_menu = menubar.addMenu("文件(&F)")
        
        # 加载模型
        self.action_load_model = QAction("加载模型...", self)
        self.action_load_model.setShortcut(QKeySequence("Ctrl+L"))
        self.action_load_model.triggered.connect(self._load_model)
        file_menu.addAction(self.action_load_model)
        
        # 导出模型
        self.action_export_model = QAction("导出模型...", self)
        self.action_export_model.setShortcut(QKeySequence("Ctrl+E"))
        self.action_export_model.triggered.connect(self._export_model)
        file_menu.addAction(self.action_export_model)
        
        file_menu.addSeparator()
        
        # 最近文件
        self.recent_menu = file_menu.addMenu("最近文件")
        self._update_recent_menu()
        
        file_menu.addSeparator()
        
        # 重启应用
        self.action_restart = QAction("重启应用(&R)", self)
        self.action_restart.setShortcut(QKeySequence("Ctrl+Shift+R"))
        self.action_restart.triggered.connect(self._request_restart)
        file_menu.addAction(self.action_restart)

        file_menu.addSeparator()

        # 退出
        self.action_exit = QAction("退出(&X)", self)
        self.action_exit.setShortcut(QKeySequence("Alt+F4"))
        self.action_exit.triggered.connect(self.close)
        file_menu.addAction(self.action_exit)
        
        # === 编辑菜单 ===
        edit_menu = menubar.addMenu("编辑(&E)")
        
        self.action_settings = QAction("设置...", self)
        self.action_settings.setShortcut(QKeySequence("Ctrl+,"))
        self.action_settings.triggered.connect(self._show_settings)
        edit_menu.addAction(self.action_settings)
        
        # === 视图菜单 ===
        view_menu = menubar.addMenu("视图(&V)")
        
        self.action_view_workflow = QAction("工作流视图", self)
        self.action_view_workflow.setShortcut(QKeySequence("Ctrl+1"))
        self.action_view_workflow.triggered.connect(lambda: self._switch_view(0))
        view_menu.addAction(self.action_view_workflow)
        
        self.action_view_model = QAction("模型视图", self)
        self.action_view_model.setShortcut(QKeySequence("Ctrl+2"))
        self.action_view_model.triggered.connect(lambda: self._switch_view(1))
        view_menu.addAction(self.action_view_model)
        
        self.action_view_preview = QAction("预览视图", self)
        self.action_view_preview.setShortcut(QKeySequence("Ctrl+3"))
        self.action_view_preview.triggered.connect(lambda: self._switch_view(2))
        view_menu.addAction(self.action_view_preview)
        
        self.action_view_training = QAction("训练视图", self)
        self.action_view_training.setShortcut(QKeySequence("Ctrl+4"))
        self.action_view_training.triggered.connect(lambda: self._switch_view(3))
        view_menu.addAction(self.action_view_training)
        
        # === 帮助菜单 ===
        help_menu = menubar.addMenu("帮助(&H)")
        
        self.action_about = QAction("关于...", self)
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
        from src.workflow.workflow import Workflow
        from src.model_builder.model_graph import ModelGraph

        # ===== 恢复工作流 =====
        try:
            last_workflow_path = config.get('session.last_workflow_path')
            if last_workflow_path and os.path.exists(last_workflow_path):
                workflow = Workflow.load(last_workflow_path)
                if workflow:
                    # 记录文件路径（供 UI 显示文件名）
                    workflow._file_path = last_workflow_path  # type: ignore[attr-defined]
                    # 同步 main_window 内部状态 + view + controller
                    try:
                        self.main_window._workflow = workflow
                    except Exception:
                        pass
                    self.main_window._workflow_view.set_workflow(workflow)
                    try:
                        self.main_window._workflow_controller.set_workflow(workflow)
                    except Exception:
                        pass
                    try:
                        self.main_window._update_workflow_status()
                    except Exception:
                        pass
                    try:
                        config.add_recent_file(last_workflow_path)
                    except Exception:
                        pass
        except Exception:
            # 启动恢复失败不应阻断应用启动
            pass

        # ===== 恢复模型编辑器 =====
        try:
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
            "加载模型",
            "",
            "Keras模型 (*.h5 *.keras);;模型定义 (*.model.json);;SavedModel (*);;所有文件 (*)"
        )
        if file_path:
            try:
                from tensorflow import keras
                self.current_model = keras.models.load_model(file_path)
                self.main_window.set_current_model(self.current_model)
                QMessageBox.information(self, "成功", f"模型已加载: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载模型失败: {e}")
    
    def _export_model(self):
        """导出模型"""
        dialog = ExportModelDialog(self, self.current_model)
        dialog.exec()
    
    def _update_recent_menu(self):
        """更新最近文件菜单"""
        self.recent_menu.clear()
        recent_files = config.get_recent_files()
        
        if not recent_files:
            action = QAction("(无最近文件)", self)
            action.setEnabled(False)
            self.recent_menu.addAction(action)
            return
        
        for file_path in recent_files:
            action = QAction(file_path, self)
            action.triggered.connect(lambda checked, p=file_path: self._open_recent_file(p))
            self.recent_menu.addAction(action)
        
        self.recent_menu.addSeparator()
        clear_action = QAction("清除最近文件", self)
        clear_action.triggered.connect(self._clear_recent_files)
        self.recent_menu.addAction(clear_action)
    
    def _open_recent_file(self, file_path: str):
        """打开最近文件"""
        import os
        if os.path.exists(file_path):
            # 根据文件类型打开
            if file_path.endswith('.json'):
                # 工作流文件，加载到工作流视图
                from src.workflow.workflow import Workflow
                workflow = Workflow.load(file_path)
                if workflow:
                    workflow._file_path = file_path  # type: ignore[attr-defined]
                    # 同步 main_window 内部状态 + view + controller
                    try:
                        self.main_window._workflow = workflow
                    except Exception:
                        pass
                    self.main_window._workflow_view.set_workflow(workflow)
                    try:
                        self.main_window._workflow_controller.set_workflow(workflow)
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
            elif file_path.endswith(('.h5', '.keras', '.model.json')):
                # 模型文件
                self._load_model()
        else:
            QMessageBox.warning(self, "警告", f"文件不存在: {file_path}")
    
    def _clear_recent_files(self):
        """清除最近文件"""
        config.clear_recent_files()
        self._update_recent_menu()
    
    # === 视图操作 ===
    
    def _switch_view(self, index: int):
        """切换视图"""
        self.main_window._view_stack.setCurrentIndex(index)
        # 更新对应的视图按钮状态
        buttons = [
            self.main_window._workflow_btn,
            self.main_window._model_btn,
            self.main_window._preview_btn,
            self.main_window._training_btn
        ]
        if 0 <= index < len(buttons):
            buttons[index].setChecked(True)
    
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
        # 更新音频设置
        for key, value in settings.items():
            config.set(key, value, save_immediately=False)
        config.save()
        
        QMessageBox.information(self, "提示", "设置已保存，部分设置需要重启生效。")
    
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
                "确认重启",
                "检测到工作流/训练正在运行。\n"
                "重启将先停止当前运行，再启动新实例。\n\n"
                "是否继续？",
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
                "重启失败",
                message + "\n\n请查看 logs 目录下最新日志。",
            )
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        # 保存窗口状态
        config.save()
        event.accept()
