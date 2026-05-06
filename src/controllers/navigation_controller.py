# -*- coding: utf-8 -*-
"""
View Navigation Controller

Responsible for navigation and switching logic between views within the application.
"""

import logging
from typing import Optional

from PySide6.QtCore import QObject, Signal

from src.core.event_bus import get_event_bus
from src.ui.i18n import tr_
from src.controllers.view_types import ViewType
from src.controllers.navigation_dto import (
    MessageBoxSpec,
    NodeDoubleClickContext,
    NodeDoubleClickDecision,
)
logger = logging.getLogger(__name__)

class NavigationController(QObject):
    """
    视图导航控制器
    
    职责:
    - 管理视图切换逻辑
    - 处理节点双击跳转
    - 维护导航历史
    """
    
    # 控制器信号
    view_changed = Signal(int)                   # view_index
    preview_updated = Signal(str, str, dict)     # node_id, node_name, outputs
    
    def __init__(self, parent=None):
        """
        初始化导航控制器
        
        Args:
            parent: 父QObject
        """
        super().__init__(parent)
        self._event_bus = get_event_bus()
        self._current_view = ViewType.WORKFLOW
        self._selected_node_id: Optional[str] = None
        self._navigation_history: list = []
        
        self._connect_event_bus()
    
    def _connect_event_bus(self):
        """连接事件总线信号"""
        self._event_bus.node_selected.connect(self._on_node_selected)
        self._event_bus.preview_requested.connect(self._on_preview_requested)
        self._event_bus.view_switch_requested.connect(self.switch_to_view)
    
    @property
    def current_view(self) -> ViewType:
        """获取当前视图类型"""
        return self._current_view
    
    @property
    def selected_node_id(self) -> Optional[str]:
        """获取当前选中的节点ID"""
        return self._selected_node_id
    
    def switch_to_view(self, view_type: int):
        """
        切换到指定视图
        
        Args:
            view_type: 视图类型（ViewType枚举值或整数）
        """
        if isinstance(view_type, int):
            view_type = ViewType(view_type)
        
        if self._current_view != view_type:
            self._navigation_history.append(self._current_view)
            self._current_view = view_type
            self.view_changed.emit(view_type.value)
            
            view_names = [tr_("Workflow"), tr_("Model"), tr_("Preview"), tr_("Training")]
            self._event_bus.emit_status(
                tr_("Switched to {view} view").format(view=view_names[view_type])
            )
            
            logger.debug(f"View switched: {view_type.name}")
    
    def switch_to_workflow(self):
        """切换到工作流视图"""
        self.switch_to_view(ViewType.WORKFLOW)
    
    def switch_to_model(self):
        """切换到模型视图"""
        self.switch_to_view(ViewType.MODEL)
    
    def switch_to_preview(self):
        """切换到预览视图"""
        self.switch_to_view(ViewType.PREVIEW)
    
    def switch_to_training(self):
        """切换到训练视图"""
        self.switch_to_view(ViewType.TRAINING)
    
    def go_back(self) -> bool:
        """
        返回上一个视图
        
        Returns:
            是否成功返回
        """
        if self._navigation_history:
            prev_view = self._navigation_history.pop()
            self._current_view = prev_view
            self.view_changed.emit(prev_view.value)
            return True
        return False
    
    def sync_current_view(self, view_type: int):
        """
        同步当前视图状态（不触发信号）
        
        当外部直接切换视图（如用户点击视图按钮）时调用此方法，
        确保控制器内部状态与实际视图保持一致。
        
        Args:
            view_type: 视图类型索引
        """
        self._current_view = ViewType(view_type)
    
    def navigate_to_node_preview(self, node_id: str, node_name: str, outputs: dict):
        """
        导航到节点预览
        
        Args:
            node_id: 节点ID
            node_name: 节点名称
            outputs: 节点输出数据
        """
        self._selected_node_id = node_id
        self.switch_to_preview()
        self.preview_updated.emit(node_id, node_name, outputs)
        self._event_bus.preview_updated.emit(node_id, node_name, outputs)
    
    def handle_node_double_click(
        self,
        node_id: str,
        node_type: str,
        category: str,
        has_output: bool,
        outputs: dict = None
    ) -> bool:
        """
        处理节点双击事件
        
        根据节点类型和状态决定跳转目标视图。
        
        Args:
            node_id: 节点ID
            node_type: 节点类型
            category: 节点分类
            has_output: 是否有输出数据
            outputs: 节点输出数据
            
        Returns:
            是否处理了双击事件
        """
        # Keep this legacy API, but delegate the decision to the unified implementation.
        ctx = NodeDoubleClickContext(
            node_id=node_id,
            node_type=node_type,
            category=category,
            display_name=node_id,
            has_output=has_output,
            outputs=outputs or {},
        )
        decision = self.decide_node_double_click(ctx)
        return decision.handled

    def decide_node_double_click(self, ctx: NodeDoubleClickContext) -> NodeDoubleClickDecision:
        """
        Decide what to do when a workflow node is double-clicked.

        This is a UI-agnostic decision API. It can trigger view switching via existing
        controller methods, and returns a `NodeDoubleClickDecision` describing any
        additional UI actions the caller should perform (message box, preview payload, etc.).
        """
        # Special cases first (kept consistent with legacy MainWindow logic).
        if ctx.node_type == "save_model":
            if ctx.has_output:
                model_path = (ctx.outputs or {}).get("model_path")
                if model_path:
                    return NodeDoubleClickDecision(
                        handled=True,
                        message_box=MessageBoxSpec(
                            level="info",
                            title=tr_("Model save path"),
                            message=tr_("Model saved to:\n{path}").format(path=model_path),
                        ),
                    )
            return NodeDoubleClickDecision(handled=True, show_not_run_tip=True)

        if ctx.category == "control" and ctx.node_type == "loop":
            return NodeDoubleClickDecision(
                handled=True,
                message_box=MessageBoxSpec(
                    level="info",
                    title=tr_("Info"),
                    message=tr_(
                        "Loop node [{name}] cannot be previewed.\n"
                        "Please preview a specific processing node inside the loop."
                    ).format(name=ctx.display_name),
                ),
            )

        if ctx.node_type == "label_file":
            if ctx.has_output:
                labels = (ctx.outputs or {}).get("labels")
                if labels:
                    if isinstance(labels, (list, dict)):
                        label_count = len(labels)
                    else:
                        label_count = 1
                    return NodeDoubleClickDecision(
                        handled=True,
                        message_box=MessageBoxSpec(
                            level="info",
                            title=tr_("Label data"),
                            message=tr_(
                                "Loaded {count} label(s).\n"
                                "Label data cannot be visualized for preview."
                            ).format(count=label_count),
                        ),
                    )
            return NodeDoubleClickDecision(handled=True, show_not_run_tip=True)

        if ctx.node_type == "show_history" and ctx.has_output:
            history = (ctx.outputs or {}).get("history")
            if history is not None:
                # View should inject this to TrainingView before switching.
                self.switch_to_training()
                return NodeDoubleClickDecision(
                    handled=True,
                    switch_to=ViewType.TRAINING,
                    training_history=history,
                )

        # Training-related nodes.
        if ctx.category == "training":
            if ctx.node_type in ("trainer", "show_history"):
                self.switch_to_training()
                return NodeDoubleClickDecision(handled=True, switch_to=ViewType.TRAINING)

            # For other training nodes, if they have output, go to preview.
            if ctx.has_output and ctx.outputs:
                self._selected_node_id = ctx.node_id
                self.switch_to_preview()
                return NodeDoubleClickDecision(
                    handled=True,
                    switch_to=ViewType.PREVIEW,
                    preview_payload=(ctx.node_id, ctx.display_name, ctx.outputs),
                )
            return NodeDoubleClickDecision(handled=True, show_not_run_tip=True)

        # Control nodes: do not preview (legacy behavior).
        if ctx.category == "control":
            return NodeDoubleClickDecision(handled=False)

        # Default fallback: if it has output data, preview it; otherwise show tip.
        if not ctx.has_output:
            return NodeDoubleClickDecision(handled=True, show_not_run_tip=True)

        if ctx.outputs:
            self._selected_node_id = ctx.node_id
            self.switch_to_preview()
            return NodeDoubleClickDecision(
                handled=True,
                switch_to=ViewType.PREVIEW,
                preview_payload=(ctx.node_id, ctx.display_name, ctx.outputs),
            )

        return NodeDoubleClickDecision(handled=False)
    
    def _on_node_selected(self, node_id: str):
        """节点选中事件"""
        self._selected_node_id = node_id
        
        # 如果当前在预览视图，可以自动更新预览
        if self._current_view == ViewType.PREVIEW:
            # 通知预览视图更新（通过事件总线）
            pass
    
    def _on_preview_requested(self, node_id: str, data: dict):
        """预览请求事件"""
        self._selected_node_id = node_id
        if self._current_view != ViewType.PREVIEW:
            self.switch_to_preview()


