# -*- coding: utf-8 -*-
"""
Workflow View

Provides node editor view for designing and editing workflows.
"""

import logging
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QSplitter,
    QStackedWidget, QToolBar, QVBoxLayout, QWidget, QWidgetAction
)

from ..node_editor import NodeGraphWidget, NodePalette, PropertyPanel
from ..styles import Styles
from ...controllers.workflow_controller import WorkflowController
from ...workflow.engine import WorkflowEngine
from ...workflow.workflow import Workflow
from src.utils.config import config


logger = logging.getLogger(__name__)


class WorkflowView(QWidget):
    """
    工作流视图
    
    包含节点面板、节点画布和属性面板的完整编辑界面。
    
    Signals:
        workflow_changed: 工作流发生变化
        run_requested: 请求运行工作流
        node_selected: 节点被选中 (node_id)
    """
    
    workflow_changed = pyqtSignal()
    run_requested = pyqtSignal()
    node_selected = pyqtSignal(str)
    node_double_clicked = pyqtSignal(str)  # 节点双击信号
    continue_requested = pyqtSignal()  # 断点继续信号
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._workflow: Optional[Workflow] = None
        self._engine: Optional[WorkflowEngine] = None
        
        self._setup_ui()
        self._connect_signals()
    
    def _setup_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # 工具栏
        toolbar = self._create_toolbar()
        layout.addWidget(toolbar)
        
        # 主分割器
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)
        splitter.setStyleSheet(f"""
            QSplitter::handle {{
                background: {Styles.COLORS['surface1']};
            }}
        """)
        layout.addWidget(splitter)
        
        # 左侧：节点面板
        self._node_palette = NodePalette()
        self._node_palette.setMinimumWidth(180)
        self._node_palette.setMaximumWidth(280)
        splitter.addWidget(self._node_palette)
        
        # 中间：节点画布
        self._node_graph = NodeGraphWidget()
        splitter.addWidget(self._node_graph)
        
        # 右侧：属性面板
        self._property_panel = PropertyPanel()
        self._property_panel.setMinimumWidth(250)
        self._property_panel.setMaximumWidth(350)
        splitter.addWidget(self._property_panel)
        
        # 设置分割器比例
        splitter.setSizes([200, 600, 280])
    
    def _create_toolbar(self) -> QFrame:
        """创建工具栏（与模型视图风格一致）"""
        from PyQt6.QtWidgets import QSizePolicy
        
        toolbar_frame = QFrame()
        toolbar_frame.setFixedHeight(42)
        toolbar_frame.setStyleSheet(f"""
            QFrame {{
                background: {Styles.COLORS['surface0']};
                border-bottom: 1px solid {Styles.COLORS['surface1']};
            }}
        """)
        
        toolbar_layout = QHBoxLayout(toolbar_frame)
        toolbar_layout.setContentsMargins(12, 4, 12, 4)
        toolbar_layout.setSpacing(8)
        
        # 左侧：工作流名称
        self._workflow_name_label = QLabel("🔧 new_workflow")
        self._workflow_name_label.setStyleSheet(f"""
            font-size: 14px;
            font-weight: bold;
            color: {Styles.COLORS['text']};
        """)
        toolbar_layout.addWidget(self._workflow_name_label)
        
        toolbar_layout.addStretch()
        
        # 按钮样式
        btn_style = f"""
            QPushButton {{
                background: {Styles.COLORS['surface1']};
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                color: {Styles.COLORS['text']};
            }}
            QPushButton:hover {{
                background: {Styles.COLORS['surface2']};
            }}
        """
        
        # 新建按钮
        new_btn = QPushButton("📄 新建")
        new_btn.setStyleSheet(btn_style)
        new_btn.clicked.connect(self._on_new_workflow)
        toolbar_layout.addWidget(new_btn)
        
        # 打开按钮
        open_btn = QPushButton("📂 打开")
        open_btn.setStyleSheet(btn_style)
        open_btn.clicked.connect(self._on_open_workflow)
        toolbar_layout.addWidget(open_btn)
        
        # 保存按钮
        save_btn = QPushButton("💾 保存")
        save_btn.setStyleSheet(btn_style)
        save_btn.clicked.connect(self._on_save_workflow)
        toolbar_layout.addWidget(save_btn)
        
        # ===== 运行控制区域（使用 QStackedWidget 切换）=====
        self._run_control_stack = QStackedWidget()
        self._run_control_stack.setFixedHeight(32)
        self._run_control_stack.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._run_control_stack.setStyleSheet("QStackedWidget { background: transparent; }")
        
        # 页面0：正常模式（运行/停止按钮）
        normal_widget = QWidget()
        normal_widget.setStyleSheet("QWidget { background: transparent; }")
        normal_layout = QHBoxLayout(normal_widget)
        normal_layout.setContentsMargins(0, 0, 0, 0)
        normal_layout.setSpacing(4)
        
        self._run_btn = QPushButton("▶️ 运行")
        self._run_btn.clicked.connect(self._on_run_workflow)
        self._run_btn.setStyleSheet(btn_style)
        normal_layout.addWidget(self._run_btn)
        
        self._stop_btn = QPushButton("⏹️ 停止运行")
        self._stop_btn.clicked.connect(self._on_stop_workflow)
        self._stop_btn.setStyleSheet(btn_style)
        self._stop_btn.setEnabled(False)  # Only enabled while a workflow is running
        normal_layout.addWidget(self._stop_btn)
        
        self._run_control_stack.addWidget(normal_widget)  # index 0
        
        # 页面1：断点模式（断点标签 + 继续按钮）
        breakpoint_widget = QWidget()
        breakpoint_widget.setStyleSheet("QWidget { background: transparent; }")
        breakpoint_layout = QHBoxLayout(breakpoint_widget)
        breakpoint_layout.setContentsMargins(0, 0, 0, 0)
        breakpoint_layout.setSpacing(4)
        
        self._breakpoint_label = QLabel("🔴 暂停")
        self._breakpoint_label.setStyleSheet(f"""
            QLabel {{
                color: {Styles.COLORS['red']};
                font-weight: bold;
                padding: 6px 12px;
                background: {Styles.COLORS['surface1']};
                border-radius: 4px;
            }}
        """)
        breakpoint_layout.addWidget(self._breakpoint_label)
        
        self._continue_btn = QPushButton("⏵ 继续")
        self._continue_btn.clicked.connect(self._on_continue_clicked)
        self._continue_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Styles.COLORS['green']};
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                color: {Styles.COLORS['crust']};
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: {Styles.COLORS['teal']};
            }}
        """)
        breakpoint_layout.addWidget(self._continue_btn)
        
        self._run_control_stack.addWidget(breakpoint_widget)  # index 1
        
        # 默认显示正常模式
        self._run_control_stack.setCurrentIndex(0)
        
        toolbar_layout.addWidget(self._run_control_stack)
        
        # 适应画布按钮
        fit_btn = QPushButton("🔍 适应")
        fit_btn.setStyleSheet(btn_style)
        fit_btn.clicked.connect(lambda: self._node_graph.fit_to_selection())
        toolbar_layout.addWidget(fit_btn)
        
        # 清空画布按钮
        clear_btn = QPushButton("🗑️ 清空")
        clear_btn.setStyleSheet(btn_style)
        clear_btn.clicked.connect(self._on_clear_workflow)
        toolbar_layout.addWidget(clear_btn)
        
        return toolbar_frame
    
    def _on_continue_clicked(self):
        """继续按钮点击"""
        self.continue_requested.emit()
    
    def _connect_signals(self):
        """连接信号"""
        # 节点面板双击添加节点
        self._node_palette.node_add_requested.connect(self._on_add_node_from_palette)
        
        # 节点选中更新属性面板
        self._node_graph.node_selected.connect(self._on_node_selected)
        
        # 节点双击转发
        self._node_graph.node_double_clicked.connect(self.node_double_clicked)
        
        # 工作流变化
        self._node_graph.workflow_changed.connect(self._on_workflow_changed)
        
        # 属性变化
        self._property_panel.parameter_changed.connect(self._on_parameter_changed)
    
    def set_workflow(self, workflow: Workflow):
        """设置工作流"""
        self._workflow = workflow
        self._node_graph.set_workflow(workflow)
        self._property_panel.clear()
        self._update_workflow_name()
    
    def get_workflow(self) -> Optional[Workflow]:
        """获取工作流"""
        return self._node_graph.get_workflow()
    
    def _update_workflow_name(self):
        """更新工具栏显示的工作流名称"""
        import os
        
        workflow = self._workflow
        if workflow and hasattr(workflow, '_file_path') and workflow._file_path:
            # 显示文件名（不含扩展名）
            filename = os.path.basename(workflow._file_path)
            name = os.path.splitext(filename)[0]
        elif workflow:
            name = workflow.name if workflow.name else "new_workflow"
        else:
            name = "new_workflow"
        
        self._workflow_name_label.setText(f"🔧 {name}")
    
    def set_engine(self, engine: 'WorkflowEngine'):
        """
        设置工作流引擎（旧接口，保留兼容性）
        
        注意: 推荐使用 set_workflow_controller() 代替
        """
        self._engine = engine
    
    def set_workflow_controller(self, controller: 'WorkflowController'):
        """
        设置工作流控制器
        
        View 通过 Controller 间接访问 Engine，遵循前后端分离原则。
        
        Args:
            controller: WorkflowController 实例
        """
        self._workflow_controller = controller
        # 如果还需要直接访问 engine（兼容旧代码），通过 controller 获取
        self._engine = controller.engine
    
    def _on_add_node_from_palette(self, node_type: str):
        """从面板添加节点"""
        self._node_graph.add_node(node_type, (100, 100))
    
    def _on_node_selected(self, node_id: str):
        """节点选中处理"""
        workflow = self._node_graph.get_workflow()
        if workflow and node_id:
            node = workflow.get_node(node_id)
            if node:
                self._property_panel.set_node(node, node_id)
                self.node_selected.emit(node_id)
    
    def _on_workflow_changed(self):
        """工作流变化处理"""
        self.workflow_changed.emit()
    
    def _on_parameter_changed(self, node_id: str, param_name: str, value):
        """参数变化处理"""
        # 标记工作流已修改
        workflow = self._node_graph.get_workflow()
        if workflow:
            workflow._mark_dirty()
        self.workflow_changed.emit()
    
    def _on_new_workflow(self):
        """新建工作流"""
        self._node_graph.clear()
        self._workflow = Workflow("new_workflow")
        self._workflow._file_path = None  # 清除文件路径
        self._node_graph.set_workflow(self._workflow)
        self._property_panel.clear()
        self._update_workflow_name()
        self.workflow_changed.emit()
    
    def _on_open_workflow(self):
        """打开工作流"""
        from PyQt6.QtWidgets import QFileDialog
        import os
        
        path, _ = QFileDialog.getOpenFileName(
            self,
            "打开工作流",
            "workflows",
            "工作流文件 (*.json)"
        )
        if path:
            workflow = Workflow.load(path)
            if workflow:
                workflow._file_path = path  # 保存文件路径
                self.set_workflow(workflow)
                self.workflow_changed.emit()
                # persist last workflow path & recent list
                try:
                    config.set('session.last_workflow_path', path)
                    config.add_recent_file(path)
                except Exception:
                    pass
    
    def _on_save_workflow(self):
        """保存工作流"""
        from PyQt6.QtWidgets import QFileDialog
        import os
        
        workflow = self._node_graph.get_workflow()
        if not workflow:
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self,
            "保存工作流",
            f"workflows/{workflow.name}.json",
            "工作流文件 (*.json)"
        )
        if path:
            workflow.save(path)
            workflow._file_path = path  # 保存文件路径
            self._update_workflow_name()
            # persist last workflow path & recent list
            try:
                config.set('session.last_workflow_path', path)
                config.add_recent_file(path)
            except Exception:
                pass
    
    def _on_run_workflow(self):
        """运行工作流"""
        self.run_requested.emit()
    
    def _on_stop_workflow(self):
        """停止工作流"""
        from src.core.event_bus import get_event_bus

        # Immediate UI feedback: stopping can take time (e.g., waiting for training batch/epoch end).
        if hasattr(self, "_stop_btn"):
            self._stop_btn.setEnabled(False)
            self._stop_btn.setText("⏳ 正在停止...")
        if hasattr(self, "_run_btn"):
            self._run_btn.setEnabled(False)

        event_bus = get_event_bus()
        event_bus.emit_status("正在停止工作流...（等待当前任务收尾）")
        event_bus.training_stopped.emit()

        controller = getattr(self, "_workflow_controller", None)
        if controller is not None:
            controller.stop()
        elif self._engine:
            self._engine.stop()
    
    def _on_clear_workflow(self):
        """清空工作流"""
        from PyQt6.QtWidgets import QMessageBox
        
        reply = QMessageBox.question(
            self,
            "确认清空",
            "确定要清空当前工作流吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self._on_new_workflow()
    
    # ===== 节点执行状态管理 =====
    
    def update_node_state(self, node_id: str, state: str):
        """
        更新节点的执行状态
        
        Args:
            node_id: 节点ID
            state: 状态名称 (idle, running, completed, error, waiting)
        """
        self._node_graph.update_node_state(node_id, state)
    
    def reset_all_node_states(self):
        """重置所有节点的执行状态为 idle"""
        self._node_graph.reset_all_node_states()
    
    def highlight_node(self, node_id: str):
        """高亮显示指定节点"""
        self._node_graph.highlight_node(node_id)
    
    # ===== 断点控制 =====
    
    def show_breakpoint_mode(self, enabled: bool, node_name: str = ""):
        """
        显示/隐藏断点模式UI
        
        使用 QStackedWidget 切换：
        - index 0: 正常模式（运行/停止按钮）
        - index 1: 断点模式（断点标签 + 继续按钮）
        
        Args:
            enabled: True显示继续按钮和断点标签，False隐藏
            node_name: 断点节点名称（用于显示）
        """
        # 切换 stack 页面
        self._run_control_stack.setCurrentIndex(1 if enabled else 0)

        if enabled and node_name:
            self._breakpoint_label.setText(f"🔴 {node_name}")
        else:
            self._breakpoint_label.setText("🔴 暂停")

