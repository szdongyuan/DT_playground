# -*- coding: utf-8 -*-
"""
Training Controller

Responsible for controlling and monitoring the model training process.
"""

import logging
from typing import Any, Dict, Optional

from PyQt6.QtCore import QObject, pyqtSignal

from src.core.event_bus import get_event_bus
from src.services.training_params_service import TrainingParamsService
from src.ui.i18n import tr_
logger = logging.getLogger(__name__)


class TrainingController(QObject):
    """
    训练控制器
    
    职责:
    - 管理训练过程的生命周期
    - 处理训练进度更新
    - 协调训练事件与视图更新
    """
    
    # 控制器信号
    training_started = pyqtSignal(int)               # total_epochs
    epoch_completed = pyqtSignal(int, int, dict)     # current, total, metrics
    training_finished = pyqtSignal(bool, str)        # success, message
    
    def __init__(self, parent=None):
        """
        初始化训练控制器
        
        Args:
            parent: 父QObject
        """
        super().__init__(parent)
        self._event_bus = get_event_bus()
        self._is_training = False
        self._current_epoch = 0
        self._total_epochs = 0
        self._training_history: Dict[str, list] = {}
        
        self._connect_event_bus()
    
    def _connect_event_bus(self):
        """连接事件总线信号"""
        self._event_bus.node_progress.connect(self._on_node_progress)
    
    @property
    def is_training(self) -> bool:
        """是否正在训练"""
        return self._is_training
    
    @property
    def current_epoch(self) -> int:
        """当前epoch"""
        return self._current_epoch
    
    @property
    def total_epochs(self) -> int:
        """总epoch数"""
        return self._total_epochs
    
    @property
    def training_history(self) -> Dict[str, list]:
        """训练历史数据"""
        return self._training_history
    
    def start_training(self, total_epochs: int):
        """
        开始训练
        
        Args:
            total_epochs: 总epoch数
        """
        self._is_training = True
        self._current_epoch = 0
        self._total_epochs = total_epochs
        self._training_history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
        }
        
        self.training_started.emit(total_epochs)
        self._event_bus.training_started.emit(total_epochs)
        self._event_bus.emit_status(
            tr_("Training started (total {epochs} epochs)").format(epochs=total_epochs)
        )
        
        logger.info(f"Training started: {total_epochs} epochs")

    def start_training_for_workflow(self, workflow, default_epochs: int = 20) -> int:
        """
        Derive training parameters from a workflow and start training tracking.

        Returns:
            The derived total epochs.
        """
        total_epochs = TrainingParamsService.derive_total_epochs(
            workflow,
            default_epochs=default_epochs,
        )
        self.start_training(total_epochs)
        return total_epochs
    
    def update_epoch(self, current: int, total: int, metrics: Dict[str, Any]):
        """
        更新训练epoch
        
        Args:
            current: 当前epoch
            total: 总epoch数
            metrics: 训练指标
        """
        self._current_epoch = current
        self._total_epochs = total
        
        # 更新历史
        for key in ['loss', 'accuracy', 'val_loss', 'val_accuracy']:
            if key in metrics:
                self._training_history.setdefault(key, []).append(metrics[key])
        
        self.epoch_completed.emit(current, total, metrics)
        self._event_bus.training_epoch_completed.emit(current, total, metrics)
        
        # 更新状态
        loss = metrics.get('loss', 0)
        acc = metrics.get('accuracy', 0)
        self._event_bus.emit_status(
            f"Epoch {current}/{total} - loss: {loss:.4f}, acc: {acc:.4f}"
        )
    
    def finish_training(self, success: bool, message: str = ""):
        """
        完成训练
        
        Args:
            success: 是否成功
            message: 完成消息
        """
        self._is_training = False
        
        self.training_finished.emit(success, message)
        self._event_bus.training_finished.emit(success, message)
        
        if success:
            final_loss = self._training_history.get('loss', [0])[-1] if self._training_history.get('loss') else 0
            final_acc = self._training_history.get('accuracy', [0])[-1] if self._training_history.get('accuracy') else 0
            self._event_bus.emit_status(
                tr_("Training finished - final loss: {loss:.4f}, acc: {acc:.4f}").format(
                    loss=final_loss,
                    acc=final_acc,
                )
            )
        else:
            self._event_bus.emit_status(
                tr_("Training failed: {message}").format(message=message)
            )
        
        logger.info(f"Training finished: success={success}, message={message}")
    
    def stop_training(self):
        """停止训练"""
        if self._is_training:
            self._is_training = False
            self._event_bus.training_stopped.emit()
            self._event_bus.emit_status(tr_("Training stopped"))
            logger.info("Training stopped by user")
    
    def get_progress(self) -> float:
        """
        获取训练进度
        
        Returns:
            进度值 (0.0 ~ 1.0)
        """
        if self._total_epochs == 0:
            return 0.0
        return self._current_epoch / self._total_epochs
    
    def get_latest_metrics(self) -> Dict[str, float]:
        """
        获取最新的训练指标
        
        Returns:
            指标字典
        """
        metrics = {}
        for key, values in self._training_history.items():
            if values:
                metrics[key] = values[-1]
        return metrics
    
    def _on_node_progress(self, node_id: str, progress: float, data: dict):
        """
        处理节点进度事件
        
        当训练节点报告进度时，更新训练状态。
        """
        # 检查是否是训练相关的进度
        if 'epoch' in data or 'loss' in data:
            epoch = data.get('epoch', int(progress * self._total_epochs))
            self.update_epoch(epoch, self._total_epochs, data)


