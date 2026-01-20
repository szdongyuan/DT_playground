# -*- coding: utf-8 -*-
"""
全局事件总线

使用中介者模式解耦视图之间的通信。
"""

import logging
from typing import Any, Dict, Optional

from PyQt6.QtCore import QObject, pyqtSignal


logger = logging.getLogger(__name__)


class EventBus(QObject):
    """
    全局事件总线
    
    使用中介者模式集中管理应用内的事件通信，
    解耦视图、控制器和服务之间的直接依赖。
    
    使用规范
    --------
    1. **跨组件通信**: 使用 EventBus
       - 不同视图之间的通信（如 WorkflowView -> TrainingView）
       - Controller 与多个 View 之间的广播
       - 全局状态变更通知
       
    2. **组件内通信**: 使用 pyqtSignal
       - 同一个类内部的子组件通信
       - 父子组件之间的直接通信
       - Widget 与其内部元素的通信
    
    示例
    ----
    跨组件通信（推荐使用 EventBus）::
    
        # 在 WorkflowController 中发送事件
        self._event_bus.workflow_started.emit()
        
        # 在 TrainingView 中订阅事件
        get_event_bus().workflow_started.connect(self._on_workflow_started)
    
    组件内通信（推荐使用 pyqtSignal）::
    
        class WorkflowView(QWidget):
            # 组件内信号，供父组件订阅
            workflow_changed = pyqtSignal()
            node_selected = pyqtSignal(str)
            
            def _on_internal_change(self):
                self.workflow_changed.emit()  # 直接使用组件信号
    """
    
    _instance: Optional['EventBus'] = None
    
    # ===== 工作流事件 =====
    workflow_run_requested = pyqtSignal()                       # 请求运行工作流
    workflow_started = pyqtSignal()                             # 工作流开始执行
    workflow_finished = pyqtSignal(bool, str)                   # 执行完成 (success, message)
    workflow_error = pyqtSignal(str)                            # 执行错误 (error_message)
    workflow_saved = pyqtSignal(str)                            # 工作流已保存 (filepath)
    workflow_loaded = pyqtSignal(str)                           # 工作流已加载 (filepath)
    
    # ===== 节点事件 =====
    node_selected = pyqtSignal(str)                             # 节点被选中 (node_id)
    node_deselected = pyqtSignal()                              # 取消选中
    node_started = pyqtSignal(str)                              # 节点开始执行 (node_id)
    node_finished = pyqtSignal(str, bool)                       # 节点执行完成 (node_id, success)
    node_progress = pyqtSignal(str, float, dict)                # 节点进度 (node_id, progress, data)
    
    # ===== 断点事件 =====
    breakpoint_hit = pyqtSignal(str)                            # 触发断点 (node_id)
    breakpoint_continue = pyqtSignal()                          # 继续执行
    
    # ===== 预览事件 =====
    preview_requested = pyqtSignal(str, dict)                   # 请求预览 (node_id, outputs)
    preview_updated = pyqtSignal(str, str, dict)                # 预览更新 (node_id, node_name, outputs)
    
    # ===== 训练事件 =====
    training_started = pyqtSignal(int)                          # 训练开始 (total_epochs)
    training_epoch_completed = pyqtSignal(int, int, dict)       # Epoch完成 (current, total, metrics)
    training_finished = pyqtSignal(bool, str)                   # 训练完成 (success, message)
    training_stopped = pyqtSignal()                             # 训练被停止
    
    # ===== 模型事件 =====
    model_loaded = pyqtSignal(object)                           # 模型加载完成 (model)
    model_saved = pyqtSignal(str)                               # 模型保存完成 (filepath)
    model_built = pyqtSignal(object)                            # 模型构建完成 (model)
    
    # ===== 视图导航事件 =====
    view_switch_requested = pyqtSignal(int)                     # 请求切换视图 (view_index)
    
    # ===== 状态消息事件 =====
    status_message = pyqtSignal(str)                            # 状态栏消息
    status_message_temporary = pyqtSignal(str, int)             # 临时消息 (message, duration_ms)
    
    def __init__(self, parent=None):
        super().__init__(parent)
    
    @classmethod
    def instance(cls) -> 'EventBus':
        """
        获取全局单例实例
        
        Returns:
            EventBus 实例
        """
        if cls._instance is None:
            cls._instance = cls()
            logger.debug("EventBus 实例已创建")
        return cls._instance
    
    @classmethod
    def reset(cls):
        """
        重置单例实例（仅用于测试）
        """
        if cls._instance is not None:
            cls._instance.deleteLater()
            cls._instance = None
            logger.debug("EventBus 实例已重置")
    
    def emit_status(self, message: str):
        """
        发送状态消息的便捷方法
        
        Args:
            message: 状态消息
        """
        self.status_message.emit(message)
    
    def emit_node_progress(self, node_id: str, progress: float, **kwargs):
        """
        发送节点进度的便捷方法
        
        Args:
            node_id: 节点ID
            progress: 进度值 (0.0 ~ 1.0)
            **kwargs: 附加数据
        """
        self.node_progress.emit(node_id, progress, kwargs)
    
    def emit_training_epoch(self, current: int, total: int, **metrics):
        """
        发送训练Epoch完成的便捷方法
        
        Args:
            current: 当前epoch
            total: 总epoch数
            **metrics: 训练指标 (loss, accuracy, val_loss, val_accuracy等)
        """
        self.training_epoch_completed.emit(current, total, metrics)


def get_event_bus() -> EventBus:
    """
    获取全局事件总线实例
    
    这是 EventBus.instance() 的便捷函数。
    
    Returns:
        EventBus 实例
    """
    return EventBus.instance()


