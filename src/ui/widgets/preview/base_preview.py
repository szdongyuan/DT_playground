# -*- coding: utf-8 -*-
"""
Preview Component Abstract Base Class

Defines the common interface for all preview components.
"""

import logging
from typing import Any, List, Optional, Type

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QWidget

from src.ui.i18n import tr_
logger = logging.getLogger(__name__)


class BasePreviewWidget(QWidget):
    """
    Base class for all preview widgets.

    All specialized preview widgets should inherit from this class and implement
    `set_data` and `clear`.
    
    Signals:
        data_changed: emitted when displayed data changes
    """
    
    # 组件显示名称
    display_name: str = tr_("Base preview")
    
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
        """Return the currently displayed data object."""
        return self._current_data
    
    @property
    def data_info(self) -> str:
        """Return a short info string shown in the header."""
        return self._data_info
    
    def set_data(self, data: Any) -> bool:
        """
        Set the data to be displayed.
        
        Args:
            data: data object to display
            
        Returns:
            True if data was accepted and rendered
        
        Note:
            Subclasses must implement this method.
        """
        raise NotImplementedError(tr_("Subclasses must implement set_data()"))
    
    def clear(self):
        """
        Clear current view.
        
        Note:
            Subclasses must implement this method.
        """
        raise NotImplementedError(tr_("Subclasses must implement clear()"))
    
    @classmethod
    def can_display(cls, data: Any) -> bool:
        """
        Return whether this widget can display the given data object.
        
        Args:
            data: data object to check
            
        Returns:
            True if this widget can display the data
        """
        # Default implementation: subclasses may override for better checks.
        return False
    
    def _update_data_info(self, info: str):
        """Update the header info string."""
        self._data_info = info


# Preview widget registry.
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
    logger.debug("Registered preview widget: %s", widget_class.__name__)
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

