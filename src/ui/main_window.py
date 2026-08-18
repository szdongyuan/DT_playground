# -*- coding: utf-8 -*-
"""
Main Window Interface

Node-based workflow editor with multi-view switching support.
Uses controller pattern to separate business logic and an event bus to decouple component communication.
"""

import logging
from collections.abc import Callable
from typing import Optional

from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QMessageBox,
    QStackedWidget, QVBoxLayout, QWidget
)

from src.controllers.navigation_controller import NavigationController
from src.controllers.view_types import ViewType
from src.controllers.training_controller import TrainingController
from src.controllers.workflow_controller import WorkflowController
from src.core.event_bus import get_event_bus
from src.services.system_info_service import SystemInfoService
from src.controllers.navigation_dto import NodeDoubleClickContext
from src.ui.i18n import tr_
from src.ui.styles import Styles
from src.ui.views.model_builder_view import ModelBuilderView
from src.ui.views.preview_view import PreviewView
from src.ui.views.training_view import TrainingView
from src.ui.views.workflow_tabs_view import WorkflowTabsView
from src.workflow.engine import WorkflowEngine, ExecutionResult
from src.workflow.workflow import Workflow


logger = logging.getLogger(__name__)


class MainWindow(QWidget):
    """
    Main window widget.

    Provides multi-view switching:
    - Workflow view: node editor
    - Model view: visual model builder
    - Preview view: data visualization
    - Training view: training monitor
    """
    
    # Signals
    workflow_changed = Signal()
    training_started = Signal()
    training_stopped = Signal()
    training_completed = Signal(dict)
    view_changed = Signal(int)
    
    def __init__(
        self,
        parent=None,
        startup_progress: Callable[[int, str | None], None] | None = None,
    ):
        super().__init__(parent)
        self._startup_progress = startup_progress
        
        # Event bus
        self._event_bus = get_event_bus()

        # System/environment information service (UI-agnostic).
        self._system_info_service = SystemInfoService()
        
        # Workflow
        self._workflow: Optional[Workflow] = None
        
        # Controllers (Engine is managed inside WorkflowController)
        self._workflow_controller = WorkflowController()
        self._training_controller = TrainingController()
        self._navigation_controller = NavigationController()
        self._report_startup_progress(62, "Preparing main workspace...")
        
        # Current model (legacy compatibility)
        self.current_model = None
        
        # Currently selected node id (for preview refresh on view switch)
        self._selected_node_id: Optional[str] = None
        
        self._init_ui()
        self._report_startup_progress(82, "Preparing system status...")
        self._init_connections()
        self._init_event_bus_connections()
        self._apply_styles()
        self._start_resource_monitor()
        
        # Create default workflow
        self._create_default_workflow()
        self._report_startup_progress(88, "Main workspace ready...")

    def _report_startup_progress(self, value: int, step_text: str | None = None):
        """Forward startup milestones to the splash screen when available."""
        if self._startup_progress is None:
            return
        self._startup_progress(value, step_text)
    
    def _init_ui(self):
        """初始化界面布局"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 主视图堆栈
        self._view_stack = QStackedWidget()
        main_layout.addWidget(self._view_stack)
        
        # 创建视图
        self._workflow_view = WorkflowTabsView()
        self._report_startup_progress(68, "Loading workflow editor...")
        self._model_builder_view = ModelBuilderView()
        self._report_startup_progress(73, "Loading model builder...")
        self._preview_view = PreviewView()
        self._report_startup_progress(77, "Loading preview tools...")
        self._training_view = TrainingView()
        self._report_startup_progress(80, "Loading training monitor...")
        
        self._view_stack.addWidget(self._workflow_view)
        self._view_stack.addWidget(self._model_builder_view)
        self._view_stack.addWidget(self._preview_view)
        self._view_stack.addWidget(self._training_view)
        
        # 设置控制器到视图（View 通过 Controller 间接访问 Engine）
        self._workflow_view.set_workflow_controller(self._workflow_controller)
        
        # 状态栏
        self._create_status_bar(main_layout)
    
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
        
        self.status_label = QLabel(tr_("Ready"))
        self.status_label.setStyleSheet(f"color: {Styles.COLORS['subtext1']};")
        
        self.workflow_status = QLabel(tr_("Workflow: Unsaved"))
        self.workflow_status.setStyleSheet(f"color: {Styles.COLORS['subtext0']};")
        
        self.gpu_status = QLabel(tr_("GPU: Detecting..."))
        self.memory_status = QLabel(tr_("Memory: --"))
        
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
        
        # Detect GPU once at startup.
        self._report_startup_progress(86, "Detecting runtime environment...")
        self._detect_gpu()
    
    def _create_separator(self) -> QLabel:
        """创建分隔符"""
        sep = QLabel("|")
        sep.setStyleSheet(f"color: {Styles.COLORS['surface1']}; padding: 0 8px;")
        return sep
    
    def _detect_gpu(self):
        """Update the GPU status label."""
        info = self._system_info_service.get_gpu_info()
        if info.available:
            self.gpu_status.setText(tr_("GPU: {count} available").format(count=info.device_count))
            self.gpu_status.setStyleSheet(f"color: {Styles.COLORS['green']};")
            return

        # Not available, differentiate between "no GPU" and "unknown/error".
        if info.backend in ("tensorflow",):
            self.gpu_status.setText(tr_("GPU: CPU only"))
            self.gpu_status.setStyleSheet(f"color: {Styles.COLORS['yellow']};")
        else:
            self.gpu_status.setText(tr_("GPU: Unknown"))
            self.gpu_status.setStyleSheet(f"color: {Styles.COLORS['overlay0']};")
    
    def _start_resource_monitor(self):
        """Start periodic resource monitoring for the status bar."""
        self.resource_timer = QTimer()
        self.resource_timer.timeout.connect(self._update_resource_status)
        self.resource_timer.start(5000)
    
    def _update_resource_status(self):
        """Update memory status in the status bar."""
        info = self._system_info_service.get_memory_info()
        if info.backend == "psutil" and info.total_gb > 0:
            self.memory_status.setText(
                tr_("Memory: {used:.1f}/{total:.1f}GB").format(used=info.used_gb, total=info.total_gb)
            )
            return

        # Keep the previous value if detection is not available.
    
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
        self._workflow_view.run_from_node_requested.connect(self._on_run_from_node)
        
        # 预览视图信号
        self._preview_view.preview_requested.connect(self._on_preview_requested)
        
        # 训练视图信号
        self._training_view.pause_requested.connect(self._on_pause_training)
        self._training_view.resume_requested.connect(self._on_resume_training)
        self._training_view.stop_requested.connect(self._on_stop_training)
        self._training_view.stop_and_save_requested.connect(self._on_stop_training_and_save)
        
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
        self.view_changed.emit(view_index)
    
    def _on_preview_updated(self, node_id: str, node_name: str, outputs: dict):
        """预览数据更新"""
        self._selected_node_id = node_id
        self._preview_view.set_node_data(node_id, node_name, outputs)
    
    def _on_event_workflow_started(self):
        """EventBus: workflow started."""
        if self._workflow_view.get_running_tab_index() is None:
            self._workflow_view.set_running_tab_index(self._workflow_view.get_current_tab_index())
        self.training_started.emit()
        self._reset_runtime_node_states_for_active_run()
        # Ensure workflow run controls are updated when a run starts.
        self._workflow_view.set_run_controls_state(
            run_enabled=False,
            stop_enabled=True,
            stop_text=tr_("⏹️ Stop"),
        )
    
    def _on_event_workflow_finished(self, success: bool, message: str):
        """EventBus: workflow finished."""
        if success:
            self.training_completed.emit({})
        self._workflow_view.clear_breakpoint_mode()
        self._workflow_view.clear_running_tab()
        # Restore workflow run controls (stop may take time and temporarily disables buttons).
        self._workflow_view.set_run_controls_state(
            run_enabled=True,
            stop_enabled=False,
            stop_text=tr_("⏹️ Stop"),
        )
    
    def _on_event_workflow_error(self, error_msg: str):
        """EventBus: workflow error."""
        self._workflow_view.clear_breakpoint_mode()
        self._workflow_view.clear_running_tab()
        # Restore workflow run controls on error as well.
        self._workflow_view.set_run_controls_state(
            run_enabled=True,
            stop_enabled=False,
            stop_text=tr_("⏹️ Stop"),
        )
        QMessageBox.critical(self, tr_("Execution error"), error_msg)
    
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
        self._workflow_view.activate_running_tab()
        self._workflow_btn.setChecked(True)
        self._view_stack.setCurrentWidget(self._workflow_view)
        self._workflow_view.highlight_node(node_id)
        self._workflow_view.update_node_state(node_id, 'waiting')
        
        workflow = self._workflow_controller.workflow or self._workflow_view.get_workflow()
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
        self._report_startup_progress(87, "Preparing default workflow...")
        self._workflow = Workflow("new_workflow")
        self._workflow_view.set_workflow(self._workflow)
        self._workflow_controller.set_workflow(self._workflow)
        self._update_workflow_status()
    
    def _on_view_changed(self, index: int):
        """视图切换"""
        self._view_stack.setCurrentIndex(index)
        self.view_changed.emit(index)
        
        # 同步导航控制器状态，确保双击节点切换视图时状态一致
        self._navigation_controller.sync_current_view(index)
        
        view_names = [
            tr_("Workflow"),
            tr_("Model"),
            tr_("Preview"),
            tr_("Training"),
        ]
        self.status_label.setText(
            tr_("Switched to {view} view").format(view=view_names[index])
        )
        
        # 如果切换到预览视图，尝试更新当前选中节点的预览
        if index == 2 and self._selected_node_id:  # 索引2是预览视图
            self._update_preview(self._selected_node_id)

    def switch_view(self, index: int):
        """Switch to one of the primary application views."""
        self._on_view_changed(index)
    
    def _on_workflow_changed(self):
        """工作流变化"""
        self._update_workflow_status()
        self.workflow_changed.emit()
    
    def _update_workflow_status(self):
        """更新工作流状态显示"""
        workflow = self._workflow_view.get_workflow()
        if workflow:
            status = tr_("Modified") if workflow.is_dirty else tr_("Saved")
            self.workflow_status.setText(
                tr_("Workflow: {name} ({status})").format(
                    name=workflow.name, status=status
                )
            )
        else:
            self.workflow_status.setText(tr_("Workflow: None"))
    
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
        outputs = node.get_preview_outputs()
        
        self._preview_view.set_node_data(node_id, node.display_name, outputs)
    
    def _on_run_workflow(self):
        """Handle workflow run request from the UI."""
        if self._is_workflow_run_active():
            running_idx = self._workflow_view.get_running_tab_index()
            current_idx = self._workflow_view.get_current_tab_index()
            if running_idx is not None and current_idx != running_idx:
                self._event_bus.emit_status(
                    tr_("A workflow is already running. Switching to the running tab.")
                )
                self._workflow_view.activate_running_tab()
                return

            self._event_bus.emit_status(tr_("A workflow is already running."))
            return

        workflow = self._workflow_view.get_workflow()
        if not workflow:
            QMessageBox.warning(self, tr_("Warning"), tr_("No runnable workflow."))
            return

        self._workflow_view.set_running_tab_index(self._workflow_view.get_current_tab_index())
        
        # Update controller's workflow first so downstream calls have a consistent source of truth.
        self._workflow_controller.set_workflow(workflow)

        # Validate via controller (UI should not call domain validation directly).
        valid, errors = self._workflow_controller.validate_workflow(workflow)
        if not valid:
            QMessageBox.warning(
                self,
                tr_("Validation failed"),
                tr_("Workflow validation failed:\n") + "\n".join(errors),
            )
            self._workflow_view.clear_running_tab()
            return

        # 通过控制器执行工作流
        success = self._workflow_controller.run(validate=False)
        if not success:
            # Do not start training tracking if the workflow didn't actually start.
            self._workflow_view.clear_breakpoint_mode()
            self._workflow_view.clear_running_tab()
            self._workflow_view.set_run_controls_state(
                run_enabled=True,
                stop_enabled=False,
                stop_text=tr_("⏹️ Stop"),
            )
            self._training_controller.finish_training(False, tr_("Startup failed"))
            return

        # Notify training controller using derived params (UI should not traverse nodes).
        # Only start training tracking after the workflow successfully starts.
        self._training_controller.start_training_for_workflow(workflow, default_epochs=20)

    def _on_run_from_node(self, node_id: str):
        """Handle partial workflow rerun starting from the selected node."""
        if self._is_workflow_run_active():
            running_idx = self._workflow_view.get_running_tab_index()
            current_idx = self._workflow_view.get_current_tab_index()
            if running_idx is not None and current_idx != running_idx:
                self._event_bus.emit_status(
                    tr_("A workflow is already running. Switching to the running tab.")
                )
                self._workflow_view.activate_running_tab()
                return

            self._event_bus.emit_status(tr_("A workflow is already running."))
            return

        workflow = self._workflow_view.get_workflow()
        if not workflow:
            QMessageBox.warning(self, tr_("Warning"), tr_("No runnable workflow."))
            return

        node = workflow.get_node(node_id)
        if node is None:
            QMessageBox.warning(self, tr_("Warning"), tr_("Selected node does not exist."))
            return

        self._workflow_view.set_running_tab_index(self._workflow_view.get_current_tab_index())
        self._workflow_controller.set_workflow(workflow)

        valid, errors = self._workflow_controller.validate_workflow(workflow)
        if not valid:
            QMessageBox.warning(
                self,
                tr_("Validation failed"),
                tr_("Workflow validation failed:\n") + "\n".join(errors),
            )
            self._workflow_view.clear_running_tab()
            return

        success = self._workflow_controller.run_from_node(node_id, validate=False)
        if not success:
            self._workflow_view.clear_breakpoint_mode()
            self._workflow_view.clear_running_tab()
            self._workflow_view.set_run_controls_state(
                run_enabled=True,
                stop_enabled=False,
                stop_text=tr_("⏹️ Stop"),
            )
            self._training_controller.finish_training(False, tr_("Startup failed"))
            return

        self._selected_node_id = node_id
        self._training_controller.start_training_for_workflow(workflow, default_epochs=20)
    
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

    def _on_stop_training_and_save(self, checkpoint_path: str):
        """Stop training and save a checkpoint before exiting training loop."""
        try:
            self._event_bus.training_stop_with_checkpoint.emit(checkpoint_path)
        except Exception:
            # Avoid breaking stop flow if UI emits a bad path
            pass

        self._workflow_controller.stop()
        self._training_controller.stop_training()
    
    def _on_engine_started(self):
        """引擎开始执行"""
        self.status_label.setText(tr_("Running workflow..."))
        self.training_started.emit()
        
        # 重置所有节点的可视化状态
        self._reset_runtime_node_states_for_active_run()
    
    def _on_engine_finished(self, result: ExecutionResult):
        """引擎执行完成"""
        self._training_view.finish_training(
            result.success,
            tr_("Time: {seconds:.2f}s").format(seconds=result.execution_time)
        )
        
        if result.success:
            self.status_label.setText(tr_("Workflow completed"))
            self.training_completed.emit(result.node_results)
        else:
            self.status_label.setText(
                tr_("Execution failed: {message}").format(message=result.message)
            )
    
    def _on_engine_error(self, error_msg: str):
        """引擎执行错误"""
        self._training_view.finish_training(False, error_msg)
        self.status_label.setText(tr_("Error: {error}").format(error=error_msg))
        QMessageBox.critical(self, tr_("Execution error"), error_msg)
    
    # ===== Controller 信号处理 =====
    
    def _on_controller_execution_finished(self, success: bool, message: str):
        """控制器：工作流执行完成"""
        self._training_view.finish_training(success, message)
        
        if success:
            self.status_label.setText(tr_("Workflow completed"))
            self.training_completed.emit({})
        else:
            self.status_label.setText(
                tr_("Execution failed: {message}").format(message=message)
            )
    
    def _on_controller_node_state_changed(self, node_id: str, state: str):
        """控制器：节点状态变化"""
        self._workflow_view.update_node_state(node_id, state)
        
        # 如果是运行状态，更新状态栏
        if state == 'running':
            workflow = self._workflow_controller.workflow or self._workflow_view.get_workflow()
            if workflow:
                node = workflow.get_node(node_id)
                if node:
                    self.status_label.setText(
                        tr_("Running: {name}").format(name=node.display_name)
                    )

    def _is_workflow_run_active(self) -> bool:
        try:
            engine = self._workflow_controller.engine
            return engine.is_active()
        except Exception:
            return bool(self._workflow_controller.is_running() or self._workflow_controller.is_at_breakpoint())
    
    # ===== Engine 直接信号处理 (旧代码兼容，将逐步迁移) =====
    
    def _on_node_started(self, node_id: str):
        """节点开始执行"""
        workflow = self._workflow_view.get_workflow()
        if workflow:
            node = workflow.get_node(node_id)
            if node:
                self.status_label.setText(
                    tr_("Running: {name}").format(name=node.display_name)
                )
        
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
        Node internal progress update (e.g. training epoch progress).
        
        Args:
            node_id: Node ID
            progress: Progress value (0.0 ~ 1.0)
            data_json: Metrics payload in JSON format
        """
        import json
        
        # 解析 metrics 数据
        try:
            metrics = json.loads(data_json)
            if not isinstance(metrics, dict):
                metrics = {}
        except (json.JSONDecodeError, TypeError):
            metrics = {}
        
        # Do not read business parameters from workflow nodes in the UI layer.
        total_epochs = self._training_controller.total_epochs or 20

        epoch_from_metrics = metrics.get("epoch")
        if isinstance(epoch_from_metrics, (int, float)):
            current_epoch = int(epoch_from_metrics)
        else:
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
                self.status_label.setText(
                    tr_("🔴 Breakpoint paused: {name} - Click to continue").format(
                        name=node.display_name
                    )
                )
        
        # 在工作流视图显示断点模式
        self._workflow_view.show_breakpoint_mode(True, node_name)
    
    def _on_continue_from_breakpoint(self):
        """Continue from a breakpoint."""
        logger.info("User requested to continue from breakpoint")
        
        # 隐藏继续按钮
        self._workflow_view.show_breakpoint_mode(False)
        
        # 通过控制器通知引擎继续执行
        self._workflow_controller.continue_from_breakpoint()
    
    def _on_node_double_clicked(self, node_id: str):
        """
        Handle node double-click.

        The UI should only gather context and execute UI actions. Routing/decision
        logic is handled by the navigation controller.
        """
        workflow = self._workflow_view.get_workflow()
        if not workflow:
            return
        
        node = workflow.get_node(node_id)
        if not node:
            return
        
        # 检查节点是否有输出数据（判断是否已运行）
        outputs = node.get_preview_outputs()
        has_output_data = bool(outputs)
        
        ctx = NodeDoubleClickContext(
            node_id=node_id,
            node_type=node.node_type,
            category=node.category.value if hasattr(node.category, "value") else str(node.category),
            display_name=node.display_name,
            has_output=has_output_data,
            outputs=outputs,
        )

        decision = self._navigation_controller.decide_node_double_click(ctx)

        if decision.show_not_run_tip:
            self._show_not_run_tip(node)
            return

        if decision.message_box is not None:
            spec = decision.message_box
            if spec.level == "warning":
                QMessageBox.warning(self, spec.title, spec.message)
            else:
                QMessageBox.information(self, spec.title, spec.message)
            return

        if decision.training_history is not None:
            self._training_view.set_history(decision.training_history)

        if decision.preview_payload is not None:
            preview_node_id, preview_node_name, preview_outputs = decision.preview_payload
            self._selected_node_id = preview_node_id
            self._preview_view.set_node_data(preview_node_id, preview_node_name, preview_outputs)
            return
    
    def _show_not_run_tip(self, node):
        """显示节点未运行提示"""
        QMessageBox.information(
            self,
            tr_("Info"),
            tr_(
                "Node [{name}] has not been run yet.\n"
                "Click \"Run\" to execute the workflow first."
            ).format(name=node.display_name),
        )

    def _reset_runtime_node_states_for_active_run(self):
        """Reset only the nodes participating in the active run when possible."""
        workflow = self._workflow_controller.workflow or self._workflow_view.get_workflow()
        active_node_ids = self._workflow_controller.engine.get_active_run_node_ids()
        if workflow and active_node_ids and len(active_node_ids) < len(workflow.nodes):
            self._workflow_view.reset_node_states(sorted(active_node_ids))
            return

        self._workflow_view.reset_all_node_states()
    
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
