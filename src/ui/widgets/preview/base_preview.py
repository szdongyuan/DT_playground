# -*- coding: utf-8 -*-
"""
Preview Component Abstract Base Class

Defines the common interface for all preview components.
"""

import logging
from typing import Any, List, Optional, Type

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QWidget

from src.workflow.port import DataType


logger = logging.getLogger(__name__)


class BasePreviewWidget(QWidget):
    """
    预览组件基类
    
    所有专用预览组件都应继承此类并实现 set_data 和 clear 方法。
    
    Signals:
        data_changed: 当显示的数据发生变化时发射
    """
    
    # 类属性：该组件支持的数据类型列表
    supported_types: List[DataType] = []
    
    # 组件显示名称
    display_name: str = "基础预览"
    
    # 组件图标
    icon: str = "📊"
    
    # 信号
    data_changed = pyqtSignal()
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._current_data: Any = None
        self._data_info: str = ""
    
    @property
    def current_data(self) -> Any:
        """获取当前显示的数据"""
        return self._current_data
    
    @property
    def data_info(self) -> str:
        """获取数据信息字符串（用于显示在标题栏）"""
        return self._data_info
    
    def set_data(self, data: Any) -> bool:
        """
        设置要显示的数据
        
        Args:
            data: 要显示的数据对象
            
        Returns:
            是否成功设置数据
        
        Note:
            子类必须实现此方法
        """
        raise NotImplementedError("子类必须实现 set_data 方法")
    
    def clear(self):
        """
        清除当前显示
        
        Note:
            子类必须实现此方法
        """
        raise NotImplementedError("子类必须实现 clear 方法")
    
    @classmethod
    def can_display(cls, data: Any) -> bool:
        """
        检查该组件是否能显示给定的数据
        
        Args:
            data: 要检查的数据对象
            
        Returns:
            是否可以显示该数据
        """
        # 默认实现：子类可以覆盖以提供更精确的检查
        return False
    
    def _update_data_info(self, info: str):
        """更新数据信息"""
        self._data_info = info


# 预览组件注册表
_preview_registry: List[Type[BasePreviewWidget]] = []


def register_preview(widget_class: Type[BasePreviewWidget]) -> Type[BasePreviewWidget]:
    """
    预览组件注册装饰器
    
    使用方法:
        @register_preview
        class MyPreviewWidget(BasePreviewWidget):
            ...
    """
    _preview_registry.append(widget_class)
    logger.debug(f"注册预览组件: {widget_class.__name__}")
    return widget_class


def get_preview_widget_for_data(data: Any) -> Optional[Type[BasePreviewWidget]]:
    """
    根据数据类型获取合适的预览组件类
    
    Args:
        data: 要预览的数据
        
    Returns:
        适合该数据的预览组件类，如果没有找到返回 None
    """
    for widget_class in _preview_registry:
        if widget_class.can_display(data):
            return widget_class
    return None


def get_all_preview_widgets() -> List[Type[BasePreviewWidget]]:
    """获取所有已注册的预览组件类"""
    return list(_preview_registry)

