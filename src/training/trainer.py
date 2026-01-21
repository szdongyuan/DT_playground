"""
Trainer Module
Provides complete model training functionality

Architecture Notes
------------------
- Base callback classes are defined in `callbacks.py`
- `TrainerCallback` is an extended callback for `TrainerWorker` (includes total_epochs tracking)
- Node `TrainerNode` directly uses `TrainingCallback` from `callbacks.py`
"""

import os
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from PyQt6.QtCore import QObject, QThread, pyqtSignal
from tensorflow import keras

# 从 callbacks 模块导入基础回调
from .callbacks import TrainingCallback as BaseTrainingCallback


class TrainerCallback(BaseTrainingCallback):
    """
    TrainerWorker 专用回调
    
    扩展基础回调，增加 total_epochs 追踪和兼容的 epoch_callback 签名。
    """
    
    def __init__(self, 
                 progress_callback: Callable = None,
                 epoch_callback: Callable = None,
                 batch_callback: Callable = None):
        # 将 batch_callback 转换为基类格式
        super().__init__(
            on_epoch_end=None,  # 使用自定义处理
            on_batch_end=batch_callback
        )
        self.progress_callback = progress_callback
        self.epoch_callback = epoch_callback
        self.total_epochs = 0
        self.current_epoch = 0
    
    def set_total_epochs(self, epochs: int):
        """设置总epoch数"""
        self.total_epochs = epochs
    
    def on_epoch_end(self, epoch, logs=None):
        """Epoch结束回调（扩展签名）"""
        self.current_epoch = epoch + 1
        
        if self._stop_training:
            self.model.stop_training = True
            return
        
        if self.epoch_callback:
            self.epoch_callback(
                epoch + 1,
                self.total_epochs,
                logs.get('loss', 0),
                logs.get('accuracy', 0),
                logs.get('val_loss', 0),
                logs.get('val_accuracy', 0)
            )
    
    def request_stop(self):
        """请求停止训练"""
        self._stop_training = True


# 向后兼容别名
TrainingCallback = TrainerCallback


class TrainerWorker(QThread):
    """训练工作线程"""
    
    # 信号
    epoch_end = pyqtSignal(int, int, float, float, float, float)
    batch_end = pyqtSignal(int, dict)
    training_finished = pyqtSignal(dict)
    training_error = pyqtSignal(str)
    model_saved = pyqtSignal(str)
    status_update = pyqtSignal(str)
    
    def __init__(self, 
                 model: keras.Model,
                 train_data,
                 val_data,
                 config: dict):
        """
        初始化训练工作线程
        
        Args:
            model: Keras模型
            train_data: 训练数据（生成器或元组）
            val_data: 验证数据（生成器或元组）
            config: 训练配置
        """
        super().__init__()
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.config = config
        self._stop_flag = False
        self.callback = None
    
    def run(self):
        """执行训练"""
        try:
            self.status_update.emit("正在准备训练...")
            
            # 获取配置
            epochs = self.config.get('epochs', 50)
            batch_size = self.config.get('batch_size', 32)
            learning_rate = self.config.get('learning_rate', 0.001)
            optimizer_name = self.config.get('optimizer', 'Adam')
            loss_func = self.config.get('loss', 'categorical_crossentropy')
            
            # 早停配置
            early_stopping = self.config.get('early_stopping', True)
            patience = self.config.get('patience', 10)
            min_delta = self.config.get('min_delta', 0.0001)
            
            # 模型保存配置
            save_best = self.config.get('save_best', True)
            save_path = self.config.get('save_path', '')
            
            # 创建优化器
            optimizer = self._create_optimizer(optimizer_name, learning_rate)
            
            # 编译模型
            self.status_update.emit("正在编译模型...")
            self.model.compile(
                optimizer=optimizer,
                loss=loss_func,
                metrics=['accuracy']
            )
            
            # 创建回调列表
            callbacks = []
            
            # 自定义进度回调
            self.callback = TrainingCallback(
                epoch_callback=self._on_epoch_end
            )
            self.callback.set_total_epochs(epochs)
            callbacks.append(self.callback)
            
            # 早停回调
            if early_stopping:
                early_stop = keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=patience,
                    min_delta=min_delta,
                    restore_best_weights=True,
                    verbose=0
                )
                callbacks.append(early_stop)
            
            # 模型保存回调
            if save_best and save_path:
                os.makedirs(save_path, exist_ok=True)
                checkpoint_path = os.path.join(
                    save_path, 
                    f'model_best_{datetime.now().strftime("%Y%m%d_%H%M%S")}.keras'
                )
                checkpoint = keras.callbacks.ModelCheckpoint(
                    checkpoint_path,
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=0
                )
                callbacks.append(checkpoint)
            
            # 学习率衰减
            if self.config.get('lr_scheduler', False):
                lr_scheduler = keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7,
                    verbose=0
                )
                callbacks.append(lr_scheduler)
            
            self.status_update.emit("开始训练...")
            
            # 判断数据类型并训练
            if hasattr(self.train_data, '__getitem__'):
                # 数据生成器
                history = self.model.fit(
                    self.train_data,
                    validation_data=self.val_data,
                    epochs=epochs,
                    callbacks=callbacks,
                    verbose=0
                )
            else:
                # 数组数据
                x_train, y_train = self.train_data
                x_val, y_val = self.val_data
                
                history = self.model.fit(
                    x_train, y_train,
                    validation_data=(x_val, y_val),
                    batch_size=batch_size,
                    epochs=epochs,
                    callbacks=callbacks,
                    verbose=0
                )
            
            # 返回训练历史
            result = {
                'loss': history.history.get('loss', []),
                'accuracy': history.history.get('accuracy', []),
                'val_loss': history.history.get('val_loss', []),
                'val_accuracy': history.history.get('val_accuracy', []),
                'epochs_trained': len(history.history.get('loss', [])),
                'stopped_early': early_stopping and len(history.history.get('loss', [])) < epochs
            }
            
            if save_best and save_path:
                result['model_path'] = checkpoint_path
                self.model_saved.emit(checkpoint_path)
            
            self.training_finished.emit(result)
            
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.training_error.emit(error_msg)
    
    def _create_optimizer(self, name: str, learning_rate: float):
        """创建优化器"""
        optimizers = {
            'Adam': keras.optimizers.Adam,
            'SGD': keras.optimizers.SGD,
            'RMSprop': keras.optimizers.RMSprop,
            'AdamW': keras.optimizers.AdamW
        }
        
        optimizer_class = optimizers.get(name, keras.optimizers.Adam)
        return optimizer_class(learning_rate=learning_rate)
    
    def _on_epoch_end(self, epoch, total, loss, acc, val_loss, val_acc):
        """Epoch结束回调"""
        if self._stop_flag:
            if self.callback:
                self.callback.request_stop()
        else:
            self.epoch_end.emit(epoch, total, loss, acc, val_loss, val_acc)
    
    def stop(self):
        """停止训练"""
        self._stop_flag = True
        if self.callback:
            self.callback.request_stop()


class Trainer(QObject):
    """训练管理器"""
    
    # 信号
    progress_updated = pyqtSignal(int, int, float, float, float, float)
    training_completed = pyqtSignal(dict)
    training_error = pyqtSignal(str)
    status_changed = pyqtSignal(str)
    model_saved = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None
        self.model = None
        self.is_training = False
    
    def start_training(self, 
                       model: keras.Model,
                       train_data,
                       val_data,
                       config: dict):
        """
        开始训练
        
        Args:
            model: Keras模型
            train_data: 训练数据
            val_data: 验证数据
            config: 训练配置
        """
        if self.is_training:
            return
        
        self.model = model
        self.is_training = True
        
        # 创建工作线程
        self.worker = TrainerWorker(model, train_data, val_data, config)
        self.worker.epoch_end.connect(self._on_epoch_end)
        self.worker.training_finished.connect(self._on_finished)
        self.worker.training_error.connect(self._on_error)
        self.worker.status_update.connect(self._on_status)
        self.worker.model_saved.connect(self._on_model_saved)
        self.worker.start()
    
    def stop_training(self):
        """停止训练"""
        if self.worker and self.worker.isRunning():
            self.status_changed.emit("正在停止训练...")
            self.worker.stop()
            self.worker.wait(5000)  # 等待最多5秒
            self.is_training = False
            self.status_changed.emit("训练已停止")
    
    def _on_epoch_end(self, epoch, total, loss, acc, val_loss, val_acc):
        """Epoch结束回调"""
        self.progress_updated.emit(epoch, total, loss, acc, val_loss, val_acc)
    
    def _on_finished(self, history):
        """训练完成回调"""
        self.is_training = False
        self.training_completed.emit(history)
        self.status_changed.emit("训练完成")
    
    def _on_error(self, error_msg):
        """错误回调"""
        self.is_training = False
        self.training_error.emit(error_msg)
        self.status_changed.emit("训练出错")
    
    def _on_status(self, status: str):
        """状态更新回调"""
        self.status_changed.emit(status)
    
    def _on_model_saved(self, path: str):
        """模型保存回调"""
        self.model_saved.emit(path)


class TrainingPipeline:
    """
    训练流水线
    整合数据加载、特征提取、模型训练的完整流程
    """
    
    def __init__(self,
                 sample_rate: int = 44100,
                 duration: float = 3.0,
                 feature_type: str = 'mel_spectrogram',
                 n_mels: int = 128,
                 n_mfcc: int = 20):
        """初始化训练流水线"""
        self.sample_rate = sample_rate
        self.duration = duration
        self.feature_type = feature_type
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        
        self.model = None
        self.history = None
        self.class_names = []
    
    def prepare_data(self,
                     file_paths: List[str],
                     labels: List[int],
                     train_ratio: float = 0.8,
                     batch_size: int = 32):
        """
        准备训练数据
        
        Returns:
            (训练生成器, 验证生成器, 特征形状)
        """
        from src.training.data_generator import create_data_generators
        
        train_gen, val_gen = create_data_generators(
            file_paths, labels,
            train_ratio=train_ratio,
            batch_size=batch_size,
            sample_rate=self.sample_rate,
            duration=self.duration,
            feature_type=self.feature_type,
            n_mels=self.n_mels
        )
        
        return train_gen, val_gen, train_gen.get_feature_shape()
    
    def build_model(self, input_shape: Tuple, num_classes: int,
                    model_type: str = 'cnn_2d') -> keras.Model:
        """
        构建模型
        
        Args:
            input_shape: 输入形状
            num_classes: 类别数
            model_type: 模型类型 ('cnn_1d' 或 'cnn_2d')
            
        Returns:
            Keras模型
        """
        from tensorflow.keras import layers
        
        if model_type == 'cnn_1d':
            # 1D CNN 模型（用于原始波形）
            self.model = keras.Sequential([
                layers.InputLayer(input_shape=input_shape),
                layers.Conv1D(64, 3, activation='relu', padding='same'),
                layers.MaxPooling1D(2),
                layers.Conv1D(128, 3, activation='relu', padding='same'),
                layers.MaxPooling1D(2),
                layers.Conv1D(256, 3, activation='relu', padding='same'),
                layers.GlobalAveragePooling1D(),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(num_classes, activation='softmax')
            ])
        else:
            # 2D CNN 模型（用于频谱图）
            self.model = keras.Sequential([
                layers.InputLayer(input_shape=input_shape),
                layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
                layers.GlobalAveragePooling2D(),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(num_classes, activation='softmax')
            ])
        
        return self.model
    
    def train(self, 
              train_data,
              val_data,
              config: dict) -> dict:
        """
        同步训练（用于脚本/命令行）
        
        Args:
            train_data: 训练数据
            val_data: 验证数据
            config: 训练配置
            
        Returns:
            训练历史
        """
        if self.model is None:
            raise ValueError("请先构建模型")
        
        epochs = config.get('epochs', 50)
        learning_rate = config.get('learning_rate', 0.001)
        optimizer = config.get('optimizer', 'Adam')
        loss = config.get('loss', 'categorical_crossentropy')
        
        # 编译模型
        self.model.compile(
            optimizer=keras.optimizers.get({
                'class_name': optimizer,
                'config': {'learning_rate': learning_rate}
            }),
            loss=loss,
            metrics=['accuracy']
        )
        
        # 回调
        callbacks = []
        
        if config.get('early_stopping', True):
            callbacks.append(keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=config.get('patience', 10),
                restore_best_weights=True
            ))
        
        # 训练
        history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history = history.history
        return self.history
    
    def evaluate(self, test_data) -> Dict[str, float]:
        """评估模型"""
        if self.model is None:
            raise ValueError("请先训练模型")
        
        if hasattr(test_data, '__getitem__'):
            results = self.model.evaluate(test_data, verbose=0)
        else:
            x_test, y_test = test_data
            results = self.model.evaluate(x_test, y_test, verbose=0)
        
        return {
            'loss': results[0],
            'accuracy': results[1]
        }
    
    def predict(self, audio_data: np.ndarray) -> np.ndarray:
        """预测"""
        if self.model is None:
            raise ValueError("请先训练模型")
        
        return self.model.predict(audio_data, verbose=0)
    
    def save_model(self, path: str):
        """保存模型"""
        if self.model is not None:
            self.model.save(path)
    
    def load_model(self, path: str):
        """加载模型"""
        self.model = keras.models.load_model(path)
