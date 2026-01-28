# -*- coding: utf-8 -*-
"""
Main Window Interface

Node-based workflow editor with multi-view switching support.
使用控制器模式分离业务逻辑，使用事件总线解耦组件通信。
"""

import logging
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QButtonGroup, QFrame, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QStackedWidget, QToolBar, QVBoxLayout, QWidget
)

from src.controllers.navigation_controller import NavigationController, ViewType
from src.controllers.training_controller import TrainingController
from src.controllers.workflow_controller import WorkflowController
from src.core.event_bus import get_event_bus
from src.ui.styles import Styles
from src.ui.views.model_builder_view import ModelBuilderView
from src.ui.views.preview_view import PreviewView
from src.ui.views.training_view import TrainingView
from src.ui.views.workflow_view import WorkflowView
from src.workflow.engine import WorkflowEngine, ExecutionResult
from src.workflow.workflow import Workflow


logger = logging.getLogger(__name__)


class ViewButton(QPushButton):
    """视图切换按钮"""
    
    def __init__(self, text: str, icon: str = "", parent=None):
        super().__init__(f"{icon} {text}" if icon else text, parent)
        self.setCheckable(True)
        self.setMinimumWidth(100)
        self._update_style()
    
    def _update_style(self):
        self.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                color: {Styles.COLORS['subtext1']};
                font-size: 13px;
            }}
            QPushButton:hover {{
                background: {Styles.COLORS['surface0']};
                color: {Styles.COLORS['text']};
            }}
            QPushButton:checked {{
                background: {Styles.COLORS['surface1']};
                color: {Styles.COLORS['blue']};
                font-weight: bold;
            }}
        """)


class MainWindow(QWidget):
    """
    主窗口组件
    
    采用多视图切换布局，支持：
    - 工作流视图：节点编辑器
    - 模型视图：可视化模型搭建
    - 预览视图：数据可视化
    - 训练视图：训练监控
    """
    
    # 信号定义
    workflow_changed = pyqtSignal()
    training_started = pyqtSignal()
    training_stopped = pyqtSignal()
    training_completed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 事件总线
        self._event_bus = get_event_bus()
        
        # 工作流
        self._workflow: Optional[Workflow] = None
        
        # 控制器（Engine 由 WorkflowController 内部管理）
        self._workflow_controller = WorkflowController()
        self._training_controller = TrainingController()
        self._navigation_controller = NavigationController()
        
        # 当前模型（兼容旧代码）
        self.current_model = None
        
        # 当前选中的节点ID（用于预览视图切换时更新）
        self._selected_node_id: Optional[str] = None
        
        self._init_ui()
        self._init_connections()
        self._init_event_bus_connections()
        self._apply_styles()
        self._start_resource_monitor()
        
        # 创建默认工作流
        self._create_default_workflow()
    
    def _init_ui(self):
        """初始化界面布局"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 顶部视图切换栏
        self._create_view_switcher(main_layout)
        
        # 主视图堆栈
        self._view_stack = QStackedWidget()
        main_layout.addWidget(self._view_stack)
        
        # 创建视图
        self._workflow_view = WorkflowView()
        self._model_builder_view = ModelBuilderView()
        self._preview_view = PreviewView()
        self._training_view = TrainingView()
        
        self._view_stack.addWidget(self._workflow_view)
        self._view_stack.addWidget(self._model_builder_view)
        self._view_stack.addWidget(self._preview_view)
        self._view_stack.addWidget(self._training_view)
        
        # 设置控制器到视图（View 通过 Controller 间接访问 Engine）
        self._workflow_view.set_workflow_controller(self._workflow_controller)
        
        # 状态栏
        self._create_status_bar(main_layout)
    
    def _create_view_switcher(self, layout):
        """创建视图切换栏"""
        switcher_frame = QFrame()
        switcher_frame.setFixedHeight(50)
        switcher_frame.setStyleSheet(f"""
            QFrame {{
                background: {Styles.COLORS['mantle']};
                border-bottom: 1px solid {Styles.COLORS['surface0']};
            }}
        """)
        
        switcher_layout = QHBoxLayout(switcher_frame)
        switcher_layout.setContentsMargins(12, 6, 12, 6)
        switcher_layout.setSpacing(4)
        
        # Logo/标题
        title = QLabel("🎵 AI声学训练平台")
        title.setStyleSheet(f"""
            font-size: 15px;
            font-weight: bold;
            color: {Styles.COLORS['text']};
            padding-right: 20px;
        """)
        switcher_layout.addWidget(title)
        
        # 分隔线
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.VLine)
        separator.setStyleSheet(f"background: {Styles.COLORS['surface1']};")
        separator.setFixedWidth(1)
        switcher_layout.addWidget(separator)
        
        switcher_layout.addSpacing(12)
        
        # 视图按钮组
        self._view_buttons = QButtonGroup(self)
        self._view_buttons.setExclusive(True)
        
        self._workflow_btn = ViewButton("工作流", "🔧")
        self._model_btn = ViewButton("模型", "📐")
        self._preview_btn = ViewButton("预览", "📊")
        self._training_btn = ViewButton("训练", "🏋️")
        
        self._workflow_btn.setChecked(True)
        
        for i, btn in enumerate([self._workflow_btn, self._model_btn, self._preview_btn, self._training_btn]):
            self._view_buttons.addButton(btn, i)
            switcher_layout.addWidget(btn)
        
        switcher_layout.addStretch()
        
        layout.addWidget(switcher_frame)
        
        # 连接视图切换
        self._view_buttons.idClicked.connect(self._on_view_changed)
    
    def _create_status_bar(self, layout):
        """创建状态栏"""
        status_frame = QFrame()
        status_frame.setFrameShape(QFrame.Shape.StyledPanel)
        status_frame.setMaximumHeight(28)
        status_frame.setStyleSheet(f"""
            QFrame {{
                background: {Styles.COLORS['mantle']};
                border-top: 1px solid {Styles.COLORS['surface0']};
            }}
        """)
        
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(12, 2, 12, 2)
        
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet(f"color: {Styles.COLORS['subtext1']};")
        
        self.workflow_status = QLabel("工作流: 未保存")
        self.workflow_status.setStyleSheet(f"color: {Styles.COLORS['subtext0']};")
        
        self.gpu_status = QLabel("GPU: 检测中...")
        self.memory_status = QLabel("内存: --")
        
        for label in [self.gpu_status, self.memory_status]:
            label.setStyleSheet(f"color: {Styles.COLORS['subtext0']};")
        
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        status_layout.addWidget(self.workflow_status)
        status_layout.addWidget(self._create_separator())
        status_layout.addWidget(self.memory_status)
        status_layout.addWidget(self._create_separator())
        status_layout.addWidget(self.gpu_status)
        
        layout.addWidget(status_frame)
        
        # 检测GPU
        self._detect_gpu()
    
    def _create_separator(self) -> QLabel:
        """创建分隔符"""
        sep = QLabel("|")
        sep.setStyleSheet(f"color: {Styles.COLORS['surface1']}; padding: 0 8px;")
        return sep
    
    def _detect_gpu(self):
        """检测GPU状态"""
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                self.gpu_status.setText(f"GPU: {len(gpus)}个可用")
                self.gpu_status.setStyleSheet(f"color: {Styles.COLORS['green']};")
            else:
                self.gpu_status.setText("GPU: 仅CPU")
                self.gpu_status.setStyleSheet(f"color: {Styles.COLORS['yellow']};")
        except Exception:
            self.gpu_status.setText("GPU: 未知")
            self.gpu_status.setStyleSheet(f"color: {Styles.COLORS['overlay0']};")
    
    def _start_resource_monitor(self):
        """启动资源监控"""
        self.resource_timer = QTimer()
        self.resource_timer.timeout.connect(self._update_resource_status)
        self.resource_timer.start(5000)
    
    def _update_resource_status(self):
        """更新资源状态"""
        try:
            import psutil  # type: ignore[import-not-found]
            memory = psutil.virtual_memory()
            used_gb = memory.used / (1024 ** 3)
            total_gb = memory.total / (1024 ** 3)
            self.memory_status.setText(f"内存: {used_gb:.1f}/{total_gb:.1f}GB")
        except ImportError:
            pass
    
    def _init_connections(self):
        """
        初始化信号连接
        
        注意: Engine 信号现在通过 EventBus 统一处理，
        不再直接连接 Engine。参见 _init_event_bus_connections()。
        """
        # 工作流视图信号
        self._workflow_view.workflow_changed.connect(self._on_workflow_changed)
        self._workflow_view.run_requested.connect(self._on_run_workflow)
        self._workflow_view.node_selected.connect(self._on_node_selected)
        self._workflow_view.node_double_clicked.connect(self._on_node_double_clicked)
        
        # 预览视图信号
        self._preview_view.preview_requested.connect(self._on_preview_requested)
        
        # 训练视图信号
        self._training_view.pause_requested.connect(self._on_pause_training)
        self._training_view.resume_requested.connect(self._on_resume_training)
        self._training_view.stop_requested.connect(self._on_stop_training)
        
        # 工作流视图断点继续信号
        self._workflow_view.continue_requested.connect(self._on_continue_from_breakpoint)
        
        # 导航控制器信号
        self._navigation_controller.view_changed.connect(self._on_controller_view_changed)
        self._navigation_controller.preview_updated.connect(self._on_preview_updated)
        
        # 工作流控制器信号
        self._workflow_controller.execution_started.connect(self._on_engine_started)
        self._workflow_controller.execution_finished.connect(self._on_controller_execution_finished)
        self._workflow_controller.node_state_changed.connect(self._on_controller_node_state_changed)
        self._workflow_controller.breakpoint_triggered.connect(self._on_breakpoint_hit)
    
    def _init_event_bus_connections(self):
        """初始化事件总线信号连接"""
        # 状态消息
        self._event_bus.status_message.connect(self._on_status_message)
        
        # 工作流事件
        self._event_bus.workflow_started.connect(self._on_event_workflow_started)
        self._event_bus.workflow_finished.connect(self._on_event_workflow_finished)
        self._event_bus.workflow_error.connect(self._on_event_workflow_error)
        
        # 节点事件
        self._event_bus.node_started.connect(self._on_event_node_started)
        self._event_bus.node_finished.connect(self._on_event_node_finished)
        
        # 断点事件
        self._event_bus.breakpoint_hit.connect(self._on_event_breakpoint_hit)
        
        # 训练事件
        self._event_bus.training_started.connect(self._on_event_training_started)
        self._event_bus.training_epoch_completed.connect(self._on_event_training_epoch)
        self._event_bus.training_finished.connect(self._on_event_training_finished)
    
    def _on_controller_view_changed(self, view_index: int):
        """控制器请求视图切换"""
        self._view_stack.setCurrentIndex(view_index)
        buttons = [self._workflow_btn, self._model_btn, self._preview_btn, self._training_btn]
        if 0 <= view_index < len(buttons):
            buttons[view_index].setChecked(True)
    
    def _on_preview_updated(self, node_id: str, node_name: str, outputs: dict):
        """预览数据更新"""
        self._selected_node_id = node_id
        self._preview_view.set_node_data(node_id, node_name, outputs)
    
    def _on_event_workflow_started(self):
        """事件总线：工作流开始"""
        self.training_started.emit()
        self._workflow_view.reset_all_node_states()
    
    def _on_event_workflow_finished(self, success: bool, message: str):
        """事件总线：工作流完成"""
        if success:
            self.training_completed.emit({})
    
    def _on_event_workflow_error(self, error_msg: str):
        """事件总线：工作流错误"""
        QMessageBox.critical(self, "执行错误", error_msg)
    
    def _on_event_node_started(self, node_id: str):
        """事件总线：节点开始执行"""
        self._workflow_view.update_node_state(node_id, 'running')
    
    def _on_event_node_finished(self, node_id: str, success: bool):
        """事件总线：节点执行完成"""
        state = 'completed' if success else 'error'
        self._workflow_view.update_node_state(node_id, state)
    
    def _on_event_breakpoint_hit(self, node_id: str):
        """事件总线：断点触发"""
        self._selected_node_id = node_id
        self._workflow_btn.setChecked(True)
        self._view_stack.setCurrentWidget(self._workflow_view)
        self._workflow_view.highlight_node(node_id)
        self._workflow_view.update_node_state(node_id, 'waiting')
        
        workflow = self._workflow_view.get_workflow()
        node_name = ""
        if workflow:
            node = workflow.get_node(node_id)
            if node:
                node_name = node.display_name
        self._workflow_view.show_breakpoint_mode(True, node_name)
    
    def _on_event_training_started(self, total_epochs: int):
        """事件总线：训练开始"""
        self._training_view.start_training(total_epochs)
    
    def _on_event_training_epoch(self, current: int, total: int, metrics: dict):
        """事件总线：训练Epoch完成"""
        self._training_view.update_epoch(current, total, metrics)
    
    def _on_event_training_finished(self, success: bool, message: str):
        """事件总线：训练完成"""
        self._training_view.finish_training(success, message)
    
    def _create_default_workflow(self):
        """创建默认工作流"""
        self._workflow = Workflow("new_workflow")
        self._workflow_view.set_workflow(self._workflow)
        self._workflow_controller.set_workflow(self._workflow)
        self._update_workflow_status()
    
    def _on_view_changed(self, index: int):
        """视图切换"""
        self._view_stack.setCurrentIndex(index)
        
        # 同步导航控制器状态，确保双击节点切换视图时状态一致
        self._navigation_controller.sync_current_view(index)
        
        view_names = ["工作流", "模型", "预览", "训练"]
        self.status_label.setText(f"切换到{view_names[index]}视图")
        
        # 如果切换到预览视图，尝试更新当前选中节点的预览
        if index == 2 and self._selected_node_id:  # 索引2是预览视图
            self._update_preview(self._selected_node_id)
    
    def _on_workflow_changed(self):
        """工作流变化"""
        self._update_workflow_status()
        self.workflow_changed.emit()
    
    def _update_workflow_status(self):
        """更新工作流状态显示"""
        workflow = self._workflow_view.get_workflow()
        if workflow:
            status = "已修改" if workflow.is_dirty else "已保存"
            self.workflow_status.setText(f"工作流: {workflow.name} ({status})")
        else:
            self.workflow_status.setText("工作流: 无")
    
    def _on_node_selected(self, node_id: str):
        """节点选中"""
        # 保存当前选中的节点ID
        self._selected_node_id = node_id
        
        # 如果在预览模式，更新预览
        if self._view_stack.currentWidget() == self._preview_view:
            self._update_preview(node_id)
    
    def _on_preview_requested(self, node_id: str):
        """预览请求"""
        self._update_preview(node_id)
    
    def _update_preview(self, node_id: str):
        """更新预览数据"""
        workflow = self._workflow_view.get_workflow()
        if not workflow:
            return
        
        node = workflow.get_node(node_id)
        if not node:
            return
        
        # 收集输出数据
        outputs = {
            port_name: port.data
            for port_name, port in node.outputs.items()
        }
        
        self._preview_view.set_node_data(node_id, node.display_name, outputs)
    
    def _on_run_workflow(self):
        """运行工作流"""
        workflow = self._workflow_view.get_workflow()
        if not workflow:
            QMessageBox.warning(self, "警告", "没有可运行的工作流")
            return
        
        # 验证工作流
        valid, errors = workflow.validate()
        if not valid:
            QMessageBox.warning(
                self, "验证失败",
                "工作流验证失败:\n" + "\n".join(errors)
            )
            return
        
        # 更新控制器的工作流
        self._workflow_controller.set_workflow(workflow)
        
        # 尝试从工作流中找到 TrainerNode 获取 epochs 数量
        total_epochs = 20  # 默认值
        for node in workflow.nodes.values():
            if hasattr(node, 'get_parameter'):
                try:
                    epochs = node.get_parameter("epochs")
                    if epochs:
                        total_epochs = epochs
                        break
                except:
                    pass
        
        # 通知训练控制器
        self._training_controller.start_training(total_epochs)
        
        # 通过控制器执行工作流
        success = self._workflow_controller.run()
        if not success:
            self._training_controller.finish_training(False, "启动失败")
    
    def _on_pause_training(self):
        """暂停训练"""
        self._workflow_controller.pause()
    
    def _on_resume_training(self):
        """恢复训练"""
        self._workflow_controller.resume()
    
    def _on_stop_training(self):
        """停止训练"""
        self._workflow_controller.stop()
        self._training_controller.stop_training()
    
    def _on_engine_started(self):
        """引擎开始执行"""
        self.status_label.setText("工作流执行中...")
        self.training_started.emit()
        
        # 重置所有节点的可视化状态
        self._workflow_view.reset_all_node_states()
    
    def _on_engine_finished(self, result: ExecutionResult):
        """引擎执行完成"""
        self._training_view.finish_training(
            result.success,
            f"耗时: {result.execution_time:.2f}秒"
        )
        
        if result.success:
            self.status_label.setText("工作流执行完成")
            self.training_completed.emit(result.node_results)
        else:
            self.status_label.setText(f"执行失败: {result.message}")
    
    def _on_engine_error(self, error_msg: str):
        """引擎执行错误"""
        self._training_view.finish_training(False, error_msg)
        self.status_label.setText(f"错误: {error_msg}")
        QMessageBox.critical(self, "执行错误", error_msg)
    
    # ===== Controller 信号处理 =====
    
    def _on_controller_execution_finished(self, success: bool, message: str):
        """控制器：工作流执行完成"""
        self._training_view.finish_training(success, message)
        
        if success:
            self.status_label.setText("工作流执行完成")
            self.training_completed.emit({})
        else:
            self.status_label.setText(f"执行失败: {message}")
    
    def _on_controller_node_state_changed(self, node_id: str, state: str):
        """控制器：节点状态变化"""
        self._workflow_view.update_node_state(node_id, state)
        
        # 如果是运行状态，更新状态栏
        if state == 'running':
            workflow = self._workflow_view.get_workflow()
            if workflow:
                node = workflow.get_node(node_id)
                if node:
                    self.status_label.setText(f"执行: {node.display_name}")
    
    # ===== Engine 直接信号处理 (旧代码兼容，将逐步迁移) =====
    
    def _on_node_started(self, node_id: str):
        """节点开始执行"""
        workflow = self._workflow_view.get_workflow()
        if workflow:
            node = workflow.get_node(node_id)
            if node:
                self.status_label.setText(f"执行: {node.display_name}")
        
        # 更新节点可视化状态
        self._workflow_view.update_node_state(node_id, 'running')
    
    def _on_node_finished(self, node_id: str, success: bool):
        """节点执行完成"""
        # 更新节点可视化状态
        state = 'completed' if success else 'error'
        self._workflow_view.update_node_state(node_id, state)
    
    def _on_progress_updated(self, current: int, total: int, message: str):
        """工作流整体进度更新（节点级别）"""
        self.status_label.setText(message)
    
    def _on_node_progress(self, node_id: str, progress: float, data_json: str):
        """
        节点内部进度更新（如训练epoch进度）
        
        Args:
            node_id: 节点ID
            progress: 进度值 (0.0 ~ 1.0)
            data_json: JSON格式的metrics数据
        """
        import json
        
        # 解析 metrics 数据
        try:
            metrics = json.loads(data_json)
            if not isinstance(metrics, dict):
                metrics = {}
        except (json.JSONDecodeError, TypeError):
            metrics = {}
        
        # 获取 TrainerNode 的 epochs 参数
        workflow = self._workflow_view.get_workflow()
        total_epochs = 20  # 默认值
        
        if workflow:
            node = workflow.get_node(node_id)
            if node and hasattr(node, 'get_parameter'):
                try:
                    total_epochs = node.get_parameter("epochs") or 20
                except:
                    pass
        
        current_epoch = int(round(progress * total_epochs))
        
        # 更新训练视图
        self._training_view.update_epoch(current_epoch, total_epochs, metrics)
    
    def _on_status_message(self, message: str):
        """状态消息更新（显示在状态栏）"""
        self.status_label.setText(message)
    
    def _on_breakpoint_hit(self, node_id: str):
        """
        断点触发处理
        
        当工作流执行到断点节点时：
        1. 保持在工作流视图
        2. 高亮断点节点
        3. 显示继续按钮
        """
        logger.info(f"断点触发: {node_id}")
        
        # 保存当前选中的节点
        self._selected_node_id = node_id
        
        # 切换到工作流视图
        self._workflow_btn.setChecked(True)
        self._view_stack.setCurrentWidget(self._workflow_view)
        
        # 高亮断点节点
        self._workflow_view.highlight_node(node_id)
        
        # 更新节点状态为等待中
        self._workflow_view.update_node_state(node_id, 'waiting')
        
        # 获取节点名称并显示断点模式
        workflow = self._workflow_view.get_workflow()
        node_name = ""
        if workflow:
            node = workflow.get_node(node_id)
            if node:
                node_name = node.display_name
                self.status_label.setText(f"🔴 断点暂停: {node.display_name} - 点击继续执行")
        
        # 在工作流视图显示断点模式
        self._workflow_view.show_breakpoint_mode(True, node_name)
    
    def _on_continue_from_breakpoint(self):
        """从断点继续执行"""
        logger.info("用户请求从断点继续执行")
        
        # 隐藏继续按钮
        self._workflow_view.show_breakpoint_mode(False)
        
        # 通过控制器通知引擎继续执行
        self._workflow_controller.continue_from_breakpoint()
    
    def _on_node_double_clicked(self, node_id: str):
        """
        节点双击处理
        
        根据节点类型和运行状态跳转到相应视图，委托给导航控制器处理通用逻辑。
        """
        from src.workflow.node_base import NodeCategory
        from src.workflow.port import DataType
        
        workflow = self._workflow_view.get_workflow()
        if not workflow:
            return
        
        node = workflow.get_node(node_id)
        if not node:
            return
        
        # 检查节点是否有输出数据（判断是否已运行）
        has_output_data = any(
            port.data is not None 
            for port in node.outputs.values()
        )
        
        # 收集输出数据
        outputs = {
            port_name: port.data
            for port_name, port in node.outputs.items()
        }
        
        node_type = node.node_type
        category = node.category
        
        # === 特殊节点处理 ===
        
        # 训练历史节点 - 需要特殊处理设置历史数据
        if node_type == "show_history" and has_output_data:
            history_port = node.outputs.get("history")
            if history_port and history_port.data:
                self._training_view.set_history(history_port.data)
        
        # 保存模型节点 - 显示保存路径信息
        if node_type == "save_model":
            if has_output_data:
                model_path_port = node.outputs.get("model_path")
                if model_path_port and model_path_port.data:
                    QMessageBox.information(
                        self, "模型保存路径",
                        f"模型已保存到:\n{model_path_port.data}"
                    )
                    return
            self._show_not_run_tip(node)
            return
        
        # 控制流节点
        if category == NodeCategory.CONTROL and node_type == "loop":
            QMessageBox.information(
                self, "提示",
                f"循环节点 [{node.display_name}] 无法预览\n"
                "请预览循环内部的具体处理节点"
            )
            return
        
        # 标签节点
        if node_type == "label_file":
            if has_output_data:
                labels_port = node.outputs.get("labels")
                if labels_port and labels_port.data:
                    label_count = len(labels_port.data) if isinstance(labels_port.data, (list, dict)) else 1
                    QMessageBox.information(
                        self, "标签数据",
                        f"已加载 {label_count} 个标签\n"
                        "标签数据无法可视化预览"
                    )
                    return
            self._show_not_run_tip(node)
            return
        
        # === 使用导航控制器处理通用逻辑 ===
        handled = self._navigation_controller.handle_node_double_click(
            node_id=node_id,
            node_type=node_type,
            category=category.value if hasattr(category, 'value') else str(category),
            has_output=has_output_data,
            outputs=outputs
        )
        
        if handled:
            # 导航控制器已处理，如果是预览则设置数据
            if self._navigation_controller.current_view == ViewType.PREVIEW:
                self._preview_view.set_node_data(node_id, node.display_name, outputs)
            return
        
        # === 通用处理：检查是否有数据 ===
        if not has_output_data:
            self._show_not_run_tip(node)
            return
        
        # === 根据输出数据类型跳转预览 ===
        self._switch_to_preview_for_node(node_id, node)
    
    def _show_not_run_tip(self, node):
        """显示节点未运行提示"""
        QMessageBox.information(
            self, "提示",
            f"节点 [{node.display_name}] 尚未运行\n"
            "请先点击「运行」按钮执行工作流"
        )
    
    def _switch_to_preview_for_node(self, node_id: str, node):
        """切换到预览视图并显示节点数据"""
        from src.workflow.port import DataType
        
        # 切换到预览视图
        self._preview_btn.setChecked(True)
        self._view_stack.setCurrentWidget(self._preview_view)
        
        # 收集输出数据
        outputs = {
            port_name: port.data
            for port_name, port in node.outputs.items()
        }
        
        # 确定数据类型并更新状态
        primary_port = list(node.outputs.values())[0] if node.outputs else None
        if primary_port:
            data_type = primary_port.data_type
            type_desc = self._get_data_type_description(data_type)
            self.status_label.setText(f"预览 {node.display_name} - {type_desc}")
        
        # 更新预览
        self._selected_node_id = node_id
        self._preview_view.set_node_data(node_id, node.display_name, outputs)
    
    def _get_data_type_description(self, data_type) -> str:
        """获取数据类型的描述"""
        from src.workflow.port import DataType
        
        descriptions = {
            DataType.AUDIO: "音频波形",
            DataType.FEATURE_1D: "1D特征曲线",
            DataType.FEATURE_2D: "2D特征图",
            DataType.FEATURE: "特征数据",
            DataType.MODEL: "模型",
            DataType.METRICS: "评估指标",
            DataType.LABEL: "标签数据",
            DataType.ANY: "数据",
            DataType.TRIGGER: "触发信号",
        }
        return descriptions.get(data_type, "数据")
    
    # ===== 兼容旧接口 =====
    
    def get_current_model(self):
        """获取当前模型（兼容旧代码）"""
        return self.current_model
    
    def set_current_model(self, model):
        """设置当前模型（兼容旧代码）"""
        self.current_model = model
    
    # ===== 模型构建器接口 =====
    
    def get_model_builder_view(self) -> ModelBuilderView:
        """获取模型构建器视图"""
        return self._model_builder_view
    
    def get_built_model(self):
        """获取模型构建器中构建的模型"""
        return self._model_builder_view.get_keras_model()
    
    def switch_to_model_view(self):
        """切换到模型视图"""
        self._model_btn.setChecked(True)
        self._view_stack.setCurrentWidget(self._model_builder_view)
    
    def _apply_styles(self):
        """应用样式"""
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {Styles.COLORS['base']};
                color: {Styles.COLORS['text']};
                font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
                font-size: 13px;
            }}
            
            QSplitter::handle {{
                background-color: {Styles.COLORS['surface1']};
            }}
            
            QSplitter::handle:horizontal {{
                width: 2px;
            }}
            
            QSplitter::handle:vertical {{
                height: 2px;
            }}
            
            QScrollBar:vertical {{
                background-color: {Styles.COLORS['mantle']};
                width: 10px;
                border-radius: 5px;
            }}
            
            QScrollBar::handle:vertical {{
                background-color: {Styles.COLORS['surface1']};
                border-radius: 5px;
                min-height: 20px;
            }}
            
            QScrollBar::handle:vertical:hover {{
                background-color: {Styles.COLORS['surface2']};
            }}
            
            QScrollBar:horizontal {{
                background-color: {Styles.COLORS['mantle']};
                height: 10px;
                border-radius: 5px;
            }}
            
            QScrollBar::handle:horizontal {{
                background-color: {Styles.COLORS['surface1']};
                border-radius: 5px;
                min-width: 20px;
            }}
            
            QScrollBar::add-line, QScrollBar::sub-line {{
                width: 0px;
                height: 0px;
            }}
        """)
