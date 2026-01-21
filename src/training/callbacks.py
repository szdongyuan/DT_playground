"""
Training Callbacks
"""

from typing import Callable, Optional

from tensorflow import keras


class TrainingCallback(keras.callbacks.Callback):
    """自定义训练回调，用于与UI交互"""
    
    def __init__(self, 
                 on_epoch_end: Optional[Callable] = None,
                 on_batch_end: Optional[Callable] = None,
                 on_train_begin: Optional[Callable] = None,
                 on_train_end: Optional[Callable] = None):
        """
        初始化回调
        
        Args:
            on_epoch_end: Epoch结束回调函数
            on_batch_end: Batch结束回调函数
            on_train_begin: 训练开始回调函数
            on_train_end: 训练结束回调函数
        """
        super().__init__()
        self._on_epoch_end = on_epoch_end
        self._on_batch_end = on_batch_end
        self._on_train_begin = on_train_begin
        self._on_train_end = on_train_end
        self._stop_training = False
    
    def on_train_begin(self, logs=None):
        """训练开始时调用"""
        if self._on_train_begin:
            self._on_train_begin(logs)
    
    def on_train_end(self, logs=None):
        """训练结束时调用"""
        if self._on_train_end:
            self._on_train_end(logs)
    
    def on_epoch_end(self, epoch, logs=None):
        """Epoch结束时调用"""
        if self._stop_training:
            self.model.stop_training = True
            return
        
        if self._on_epoch_end:
            self._on_epoch_end(epoch, logs)
    
    def on_batch_end(self, batch, logs=None):
        """Batch结束时调用"""
        if self._stop_training:
            self.model.stop_training = True
            return
        
        if self._on_batch_end:
            self._on_batch_end(batch, logs)
    
    def stop_training(self):
        """请求停止训练"""
        self._stop_training = True


class EarlyStoppingWithUI(keras.callbacks.EarlyStopping):
    """支持UI更新的早停回调"""
    
    def __init__(self, on_stop: Optional[Callable] = None, **kwargs):
        super().__init__(**kwargs)
        self._on_stop = on_stop
    
    def on_train_end(self, logs=None):
        super().on_train_end(logs)
        if self.stopped_epoch > 0 and self._on_stop:
            self._on_stop(self.stopped_epoch)

