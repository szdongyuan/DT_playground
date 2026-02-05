# -*- coding: utf-8 -*-
"""
View Navigation Controller

Responsible for navigation and switching logic between views within the application.
"""

import logging
from enum import IntEnum
from typing import Optional

from PyQt6.QtCore import QObject, pyqtSignal

from src.core.event_bus import get_event_bus
from src.ui.i18n import tr_
logger = logging.getLogger(__name__)


class ViewType(IntEnum):
    """视图类型枚举"""
    WORKFLOW = 0      # 工作流视图
    MODEL = 1         # 模型视图
    PREVIEW = 2       # 预览视图
    TRAINING = 3      # 训练视图


class NavigationController(QObject):
    """
    视图导航控制器
    
    职责:
    - 管理视图切换逻辑
    - 处理节点双击跳转
    - 维护导航历史
    """
    
    # 控制器信号
    view_changed = pyqtSignal(int)                   # view_index
    preview_updated = pyqtSignal(str, str, dict)     # node_id, node_name, outputs
    
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
        # 训练相关节点
        if category == "training":
            if node_type == "trainer":
                self.switch_to_training()
                return True
            elif node_type == "load_model":
                # 跳转到预览视图显示模型 summary
                if has_output and outputs:
                    self._selected_node_id = node_id
                    self.switch_to_preview()
                    return True
                return False
            elif node_type == "show_history":
                self.switch_to_training()
                return True
        
        # 控制流节点通常不需要预览
        if category == "control":
            return False
        
        # 其他节点：如果有输出数据则跳转预览
        if has_output and outputs:
            self._selected_node_id = node_id
            self.switch_to_preview()
            return True
        
        return False
    
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


