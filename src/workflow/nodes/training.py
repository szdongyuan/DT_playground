# -*- coding: utf-8 -*-
"""
Training Related Nodes

Contains model loading, trainer, evaluator, and model saving nodes.
Model definition is done through the separate "Model" view for visual construction.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from ..node_base import BaseNode, NodeCategory, register_node
from ..port import DataType

logger = logging.getLogger(__name__)


@register_node
class LoadModelNode(BaseNode):
    """
    加载模型节点
    
    支持加载:
    - Keras模型文件 (.h5, .keras, SavedModel)
    - 模型编辑器定义文件 (.model.json)
    
    编译配置直接使用模型中保存的配置（来自模型定义的 OutputLayer）。
    """
    node_type = "load_model"
    display_name = "加载模型"
    category = NodeCategory.TRAINING
    description = "加载模型文件（支持Keras模型和模型编辑器定义），使用模型内置的编译配置"
    icon = "📥"
    
    def _setup_ports(self):
        self.add_output("model", DataType.MODEL, "加载的模型")
    
    def _setup_parameters(self):
        self.add_parameter(
            "model_path", "file", "",
            display_name="模型文件路径",
            description="Keras模型(.h5/.keras)或模型定义文件(.model.json)",
            file_filter="All Models (*.h5 *.keras *.model.json);;Keras Models (*.h5 *.keras);;Model Definition (*.model.json);;SavedModel (*)",
            default_directory="model"
        )
        self.add_parameter(
            "compile_model", "bool", True,
            display_name="编译模型",
            description="是否使用模型内置的编译配置自动编译（仅对.model.json生效）"
        )
    
    def execute(self) -> bool:
        model_path = self.get_parameter("model_path")
        
        if not model_path or not os.path.exists(model_path):
            self.error_message = f"模型文件不存在: {model_path}"
            return False
        
        try:
            # 根据文件类型选择加载方式
            if model_path.endswith('.model.json'):
                model = self._load_model_definition(model_path)
            else:
                model = self._load_keras_model(model_path)
            
            if model is None:
                return False
            
            self.set_output_data("model", model)
            
            # 显示模型信息
            model_name = os.path.basename(model_path)
            self.report_status(f"模型已加载: {model_name}")
            logger.info(f"模型已加载: {model_path}")
            return True
            
        except Exception as e:
            self.error_message = f"加载模型失败: {str(e)}"
            logger.exception("加载模型异常")
            return False
    
    def _load_keras_model(self, model_path: str):
        """加载Keras模型文件"""
        import tensorflow as tf
        try:
            return tf.keras.models.load_model(model_path)
        except (ValueError, Exception) as e:
            error_str = str(e)
            # 处理包含 Lambda 层的旧模型或损坏的模型
            if "Lambda" in error_str or "marshal" in error_str:
                logger.warning(f"无法加载 .keras 模型文件: {error_str}")
                
                # 尝试从对应的 .model.json 文件重新构建
                model_json_path = self._find_model_json(model_path)
                if model_json_path and os.path.exists(model_json_path):
                    logger.info(f"尝试从模型定义文件重新构建: {model_json_path}")
                    try:
                        model = self._load_model_definition(model_json_path)
                        # 保存新的 .keras 文件
                        model.save(model_path)
                        logger.info(f"已重新构建并保存模型: {model_path}")
                        return model
                    except Exception as rebuild_e:
                        logger.error(f"从模型定义文件重建失败: {rebuild_e}")
                
                raise ValueError(
                    f"无法加载模型 {model_path}。\n"
                    "该模型文件已损坏或不兼容。\n"
                    "请在模型构建器中重新构建模型并保存。"
                ) from e
            raise
    
    def _find_model_json(self, keras_path: str) -> str:
        """根据 .keras 路径查找对应的 .model.json 文件"""
        # cnn_audio_aem.model_3s.json.keras -> cnn_audio_aem.model_3s.json.model.json
        # 或 cnn_audio_aem.keras -> cnn_audio_aem.model.json
        base = keras_path
        if base.endswith('.keras'):
            base = base[:-6]
        
        # 尝试多种可能的命名模式
        candidates = [
            base + '.model.json',
            base.replace('.json', '') + '.model.json',
            os.path.splitext(base)[0] + '.model.json',
        ]
        
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        
        return None
    
    def _load_model_definition(self, model_path: str):
        """
        加载模型定义文件并构建模型
        
        编译配置直接使用模型内置的 CompileConfig（来自 OutputLayer 或 ModelGraph）。
        """
        from src.model_builder.model_graph import ModelGraph
        
        # 加载模型图
        model_graph = ModelGraph.load(model_path)
        
        # 是否编译由参数控制，编译配置使用模型内置的
        compile_model = self.get_parameter("compile_model")
        
        # 构建 Keras 模型（编译配置在 build_keras_model 内部处理）
        model = model_graph.build_keras_model(compile_model=compile_model)
        
        if compile_model:
            config = model_graph.get_compile_config_from_output()
            logger.info(f"模型已编译（使用模型内置配置）: optimizer={config.get('optimizer')}, loss={config.get('loss')}")
        
        return model


@register_node
class SaveModelNode(BaseNode):
    """保存模型节点"""
    node_type = "save_model"
    display_name = "保存模型"
    category = NodeCategory.TRAINING
    description = "保存训练好的模型到文件"
    icon = "📤"
    
    def _setup_ports(self):
        self.add_input("model", DataType.MODEL, "待保存模型")
        self.add_output("model_path", DataType.ANY, "保存路径")
    
    def _setup_parameters(self):
        self.add_parameter(
            "save_mode", "choice", "new_file",
            display_name="保存模式",
            choices=["new_file", "overwrite"],
            description="new_file: 选择目录并输入文件名; overwrite: 选择已有文件覆盖"
        )
        self.add_parameter(
            "save_dir", "folder", "",
            display_name="保存目录",
            description="选择保存模型的目录（新建文件模式）",
            default_directory="model"
        )
        self.add_parameter(
            "file_name", "str", "model.keras",
            display_name="文件名",
            description="模型文件名（包含扩展名 .keras/.h5）"
        )
        self.add_parameter(
            "existing_file", "file", "",
            display_name="覆盖文件",
            description="选择要覆盖的模型文件（覆盖模式）",
            file_filter="Keras Models (*.h5 *.keras);;All Files (*)",
            default_directory="model"
        )
        self.add_parameter(
            "save_format", "choice", "keras",
            display_name="保存格式",
            choices=["keras", "h5", "saved_model"],
            description="模型保存格式"
        )
        self.add_parameter(
            "confirm_overwrite", "bool", True,
            display_name="覆盖警告",
            description="覆盖已有文件时是否输出警告日志"
        )
    
    def execute(self) -> bool:
        model = self.get_input_data("model")
        
        if model is None:
            self.error_message = "未提供模型"
            return False
        
        save_mode = self.get_parameter("save_mode")
        save_format = self.get_parameter("save_format")
        
        # 确定保存路径
        if save_mode == "new_file":
            save_dir = self.get_parameter("save_dir")
            file_name = self.get_parameter("file_name")
            
            if not save_dir:
                self.error_message = "未指定保存目录"
                return False
            
            if not file_name:
                self.error_message = "未指定文件名"
                return False
            
            # 确保文件名有正确的扩展名
            if save_format == "keras" and not file_name.endswith('.keras'):
                if not file_name.endswith('.h5'):
                    file_name += '.keras'
            elif save_format == "h5" and not file_name.endswith('.h5'):
                if not file_name.endswith('.keras'):
                    file_name += '.h5'
            
            save_path = os.path.join(save_dir, file_name)
            
        else:  # overwrite mode
            save_path = self.get_parameter("existing_file")
            
            if not save_path:
                self.error_message = "未选择要覆盖的文件"
                return False
        
        try:
            # 检查文件是否存在，如果存在则需要确认覆盖
            # 注意：工作流执行在后台线程中，不能在后台线程中显示 GUI 对话框
            # 这里只记录日志，直接覆盖文件
            if os.path.exists(save_path):
                if self.get_parameter("confirm_overwrite"):
                    logger.warning(f"文件已存在，将覆盖: {save_path}")
                else:
                    logger.info(f"覆盖已有文件: {save_path}")
            
            # 确保目录存在
            dir_path = os.path.dirname(save_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            
            # 保存模型
            if save_format == "saved_model":
                model.save(save_path, save_format='tf')
            else:
                model.save(save_path)
            
            self.set_output_data("model_path", save_path)
            
            # 显示保存信息
            file_name = os.path.basename(save_path)
            self.report_status(f"模型已保存: {file_name}")
            logger.info(f"模型已保存到: {save_path}")
            return True
            
        except Exception as e:
            self.error_message = f"保存模型失败: {str(e)}"
            logger.exception("保存模型异常")
            return False


@register_node
class TrainerNode(BaseNode):
    """
    训练器节点
    
    支持两种模式：
    - 使用模型内置配置（use_model_config=True）：直接使用模型中已编译的配置
    - 覆盖配置（use_model_config=False）：使用节点参数中指定的配置
    """
    node_type = "trainer"
    display_name = "训练器"
    category = NodeCategory.TRAINING
    description = "执行模型训练，支持使用模型配置或自定义覆盖"
    icon = "🏋️"
    
    def _setup_ports(self):
        self.add_input("model", DataType.MODEL, "模型")
        self.add_input("x_train", DataType.ANY, "训练数据")  # ANY类型支持特征或预处理后的音频
        self.add_input("y_train", DataType.ANY, "训练目标")  # ANY类型支持自编码器
        self.add_input("x_val", DataType.ANY, "验证数据", required=False)  # ANY类型支持特征或预处理后的音频
        self.add_input("y_val", DataType.ANY, "验证目标", required=False)  # ANY类型支持自编码器
        
        self.add_output("history", DataType.ANY, "训练历史")
        self.add_output("trained_model", DataType.MODEL, "训练后模型")
    
    def _setup_parameters(self):
        from src.model_builder.model_graph import CompileConfig
        
        self.add_parameter(
            "epochs", "int", 50,
            display_name="训练轮数",
            min_value=1
        )
        self.add_parameter(
            "batch_size", "int", 32,
            display_name="批次大小",
            min_value=1
        )
        self.add_parameter(
            "use_model_config", "bool", True,
            display_name="使用模型配置",
            description="True: 使用模型内置的编译配置; False: 使用下方的自定义配置覆盖"
        )
        # 以下参数仅在 use_model_config=False 时生效
        self.add_parameter(
            "optimizer", "choice", "Adam",
            display_name="优化器（覆盖）",
            choices=CompileConfig.OPTIMIZERS,
            description="仅当'使用模型配置'为False时生效"
        )
        self.add_parameter(
            "learning_rate", "float", 0.001,
            display_name="学习率（覆盖）",
            min_value=1e-6,
            max_value=1.0,
            description="仅当'使用模型配置'为False时生效"
        )
        self.add_parameter(
            "loss", "choice", "auto",
            display_name="损失函数（覆盖）",
            choices=["auto"] + CompileConfig.LOSSES,
            description="auto: 根据数据自动检测; 仅当'使用模型配置'为False时生效"
        )
        self.add_parameter(
            "early_stopping", "bool", True,
            display_name="早停"
        )
        self.add_parameter(
            "patience", "int", 10,
            display_name="早停耐心值",
            min_value=1
        )
    
    def execute(self) -> bool:
        try:
            import numpy as np
            from tensorflow import keras
            from .data_source import AudioData
            from src.model_builder.model_graph import CompileConfig
            from src.core.event_bus import get_event_bus
            import os
            
            model = self.get_input_data("model")
            x_train = self.get_input_data("x_train")
            y_train = self.get_input_data("y_train")
            x_val = self.get_input_data("x_val")
            y_val = self.get_input_data("y_val")
            
            if model is None:
                self.error_message = "未提供模型"
                return False
            
            if x_train is None or y_train is None:
                self.error_message = "未提供训练数据"
                return False
            
            # 转换为numpy数组（支持AudioData列表或特征列表）
            # 传入 model 参数，根据模型类型自动调整数据格式（NCHW -> NHWC for Keras）
            X = self._convert_to_array(x_train, model)
            Y = self._convert_to_array(y_train)  # 标签数据不需要转换格式
            
            # 准备验证数据
            validation_data = None
            if x_val is not None and y_val is not None:
                X_val = self._convert_to_array(x_val, model)
                Y_val = self._convert_to_array(y_val)  # 标签数据不需要转换格式
                validation_data = (X_val, Y_val)
            
            # 根据配置模式编译模型
            use_model_config = self.get_parameter("use_model_config")
            
            if use_model_config:
                # 使用模型内置配置，检查模型是否已编译
                if not model.compiled:
                    self.error_message = "模型未编译，请确保模型已包含编译配置或关闭'使用模型配置'"
                    return False
                # 获取已编译模型的配置信息用于日志
                optimizer_config = model.optimizer.get_config() if model.optimizer else {}
                optimizer_name = optimizer_config.get('name', 'unknown')
                loss_name = model.loss if isinstance(model.loss, str) else getattr(model.loss, '__name__', str(model.loss))
                self.report_status(f"使用模型内置配置: optimizer={optimizer_name}, loss={loss_name}")
            else:
                # 覆盖配置：使用节点参数
                optimizer_name = self.get_parameter("optimizer")
                learning_rate = self.get_parameter("learning_rate")
                loss_param = self.get_parameter("loss")
                
                # 使用 CompileConfig 的统一方法
                optimizer = CompileConfig.create_optimizer(optimizer_name, learning_rate)
                
                # 处理 auto 损失函数
                if loss_param == "auto":
                    loss = CompileConfig.auto_detect_loss(Y)
                else:
                    loss = loss_param
                
                metrics = CompileConfig.get_default_metrics(loss)
                
                model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
                self.report_status(f"使用覆盖配置: optimizer={optimizer_name}, loss={loss}")
            
            # 回调
            callbacks = []
            total_epochs = self.get_parameter("epochs")
            
            # 添加进度回调
            from src.training.callbacks import TrainingCallback
            
            def on_epoch_end(epoch, logs):
                # epoch 是从 0 开始的
                current_epoch = epoch + 1
                progress = current_epoch / total_epochs
                epoch_metrics = logs or {}
                self.report_progress(progress, f"Epoch {current_epoch}/{total_epochs}", epoch_metrics)
            
            class _TrainingStopAndSaveCallback(TrainingCallback):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self._checkpoint_path: Optional[str] = None

                def set_checkpoint_path(self, path: str):
                    self._checkpoint_path = path

                def on_train_end(self, logs=None):
                    super().on_train_end(logs)
                    if not self._checkpoint_path:
                        return
                    try:
                        dir_path = os.path.dirname(self._checkpoint_path)
                        if dir_path:
                            os.makedirs(dir_path, exist_ok=True)
                        # Save full model checkpoint (can be large).
                        self.model.save(self._checkpoint_path)
                    except Exception:
                        # Do not raise from callback; training is already stopping.
                        pass

            training_callback = _TrainingStopAndSaveCallback(on_epoch_end=on_epoch_end)
            callbacks.append(training_callback)

            event_bus = get_event_bus()

            def _on_training_stopped():
                training_callback.stop_training()

            def _on_training_stop_with_checkpoint(path: str):
                if path:
                    training_callback.set_checkpoint_path(path)
                training_callback.stop_training()

            event_bus.training_stopped.connect(_on_training_stopped)
            event_bus.training_stop_with_checkpoint.connect(_on_training_stop_with_checkpoint)
            
            if self.get_parameter("early_stopping"):
                callbacks.append(keras.callbacks.EarlyStopping(
                    monitor='val_loss' if validation_data else 'loss',
                    patience=self.get_parameter("patience"),
                    restore_best_weights=True
                ))
            
            # 训练
            self.report_status(f"开始训练: {total_epochs} epochs, batch_size={self.get_parameter('batch_size')}")

            try:
                history = model.fit(
                    X, Y,
                    epochs=total_epochs,
                    batch_size=self.get_parameter("batch_size"),
                    validation_data=validation_data,
                    callbacks=callbacks,
                    verbose=1
                )
            finally:
                try:
                    event_bus.training_stopped.disconnect(_on_training_stopped)
                except TypeError:
                    pass
                try:
                    event_bus.training_stop_with_checkpoint.disconnect(_on_training_stop_with_checkpoint)
                except TypeError:
                    pass
            
            self.set_output_data("trained_model", model)
            self.set_output_data("history", history.history)
            
            # 获取最终指标
            final_loss = history.history.get('loss', [0])[-1]
            final_val_loss = history.history.get('val_loss', [0])[-1] if 'val_loss' in history.history else None
            
            if final_val_loss:
                self.report_status(f"训练完成: loss={final_loss:.4f}, val_loss={final_val_loss:.4f}")
            else:
                self.report_status(f"训练完成: loss={final_loss:.4f}")
            
            logger.info("模型训练完成")
            return True
            
        except Exception as e:
            self.error_message = f"训练失败: {str(e)}"
            logger.exception("训练异常")
            return False
    
    def _convert_to_array(self, data, model=None):
        """
        将输入数据转换为numpy数组，并根据模型类型调整数据格式
        
        支持:
        - 特征列表 (List[np.ndarray])
        - AudioData列表 (List[AudioData]) - 提取波形数据
        - FeatureData列表 (List[FeatureData]) - 提取特征数据
        - 已经是numpy数组
        
        Args:
            data: 输入数据
            model: 模型对象，用于判断是否需要转换数据格式
            
        Returns:
            转换后的numpy数组
        """
        import numpy as np
        from .data_source import AudioData
        from .feature import FeatureData
        
        if isinstance(data, np.ndarray):
            arr = data
        elif isinstance(data, list) and len(data) > 0:
            # 检查是否是AudioData列表
            if isinstance(data[0], AudioData):
                # 从AudioData中提取波形数据
                # AudioData.data 格式为 (channels, samples)
                waveforms = [audio.data for audio in data]
                arr = np.array(waveforms)  # (batch, channels, samples)
            # 检查是否是FeatureData列表
            elif isinstance(data[0], FeatureData):
                # 从FeatureData中提取特征数据
                # FeatureData.data 格式为 (channels, features, frames)
                features = [feat.data for feat in data]
                arr = np.array(features)  # (batch, channels, features, frames)
            else:
                # 假设是特征列表
                arr = np.array(data)
        else:
            arr = np.array(data)
        
        # 根据模型类型调整数据格式
        arr = self._adjust_data_format(arr, model)
        
        return arr
    
    def _adjust_data_format(self, data, model):
        """
        根据模型类型调整数据格式
        
        我们的数据默认为 NCHW (channels_first) 格式：
        - 2D 数据: (batch, channels, length)
        - 3D 数据: (batch, channels, height, width)
        
        Keras 模型默认使用 NHWC (channels_last) 格式：
        - 2D 数据: (batch, length, channels)
        - 3D 数据: (batch, height, width, channels)
        
        Args:
            data: numpy数组
            model: 模型对象
            
        Returns:
            调整格式后的numpy数组
        """
        import numpy as np
        
        if model is None:
            return data
        
        # 检查是否为 Keras 模型
        if self._is_keras_model(model):
            # Keras 模型需要 channels_last 格式
            data = self._convert_to_channels_last(data)
            logger.debug(f"数据已转换为 channels_last 格式: {data.shape}")
        # 如果未来支持 PyTorch 模型，保持 NCHW 格式不变
        
        return data
    
    def _is_keras_model(self, model) -> bool:
        """判断是否为 Keras 模型"""
        try:
            from tensorflow import keras
            return isinstance(model, keras.Model)
        except ImportError:
            return False
    
    def _convert_to_channels_last(self, data):
        """
        将数据从 channels_first (NCHW) 转换为 channels_last (NHWC)
        
        转换规则：
        - 3D 数据 (batch, channels, length) -> (batch, length, channels)
        - 4D 数据 (batch, channels, height, width) -> (batch, height, width, channels)
        
        判断是否需要转换的启发式规则：
        - 对于 3D 数据：如果 axis=1 的维度明显小于 axis=2，则认为是 channels_first
        - 对于 4D 数据：如果 axis=1 的维度明显小于 axis=2 和 axis=3，则认为是 channels_first
        """
        import numpy as np
        
        ndim = data.ndim
        
        if ndim == 3:
            # (batch, channels, length) -> (batch, length, channels)
            # 启发式判断：channels 通常很小（1-16），length 通常很大（>100）
            if data.shape[1] < data.shape[2]:
                return np.transpose(data, (0, 2, 1))
        
        elif ndim == 4:
            # (batch, channels, height, width) -> (batch, height, width, channels)
            # 启发式判断：channels 通常很小（1-16），height/width 通常较大
            if data.shape[1] < data.shape[2] and data.shape[1] < data.shape[3]:
                return np.transpose(data, (0, 2, 3, 1))
        
        # 1D 或 2D 数据不需要转换，或者已经是 channels_last 格式
        return data


@register_node
class EvaluatorNode(BaseNode):
    """评估器节点"""
    node_type = "evaluator"
    display_name = "评估器"
    category = NodeCategory.TRAINING
    description = "评估模型性能"
    icon = "📊"
    
    def _setup_ports(self):
        self.add_input("model", DataType.MODEL, "模型")
        self.add_input("x_test", DataType.ANY, "测试数据")  # ANY类型支持特征或预处理后的音频
        self.add_input("y_test", DataType.ANY, "测试目标")  # ANY类型支持自编码器
        
        self.add_output("metrics", DataType.METRICS, "评估指标")
    
    def _setup_parameters(self):
        self.add_parameter(
            "batch_size", "int", 32,
            display_name="批次大小",
            min_value=1
        )
    
    def execute(self) -> bool:
        try:
            import numpy as np
            from .data_source import AudioData
            
            model = self.get_input_data("model")
            x_test = self.get_input_data("x_test")
            y_test = self.get_input_data("y_test")
            
            if model is None:
                self.error_message = "未提供模型"
                return False
            
            if x_test is None or y_test is None:
                self.error_message = "未提供测试数据"
                return False
            
            # 转换为numpy数组，传入 model 参数根据模型类型自动调整数据格式
            X = self._convert_to_array(x_test, model)
            Y = self._convert_to_array(y_test)  # 标签数据不需要转换格式
            
            # 评估
            results = model.evaluate(
                X, Y,
                batch_size=self.get_parameter("batch_size"),
                verbose=0
            )
            
            # 获取指标名称
            metric_names = model.metrics_names
            
            metrics = {}
            if isinstance(results, list):
                for name, value in zip(metric_names, results):
                    metrics[name] = float(value)
            else:
                metrics[metric_names[0]] = float(results)
            
            self.set_output_data("metrics", metrics)
            
            # 显示评估结果
            loss_str = f"loss={metrics.get('loss', 0):.4f}"
            self.report_status(f"评估完成: {loss_str}")
            
            logger.info(f"评估完成: {metrics}")
            return True
            
        except Exception as e:
            self.error_message = f"评估失败: {str(e)}"
            logger.exception("评估异常")
            return False
    
    def _convert_to_array(self, data, model=None):
        """
        将输入数据转换为numpy数组，并根据模型类型调整数据格式
        
        支持:
        - 特征列表 (List[np.ndarray])
        - AudioData列表 (List[AudioData]) - 提取波形数据
        - FeatureData列表 (List[FeatureData]) - 提取特征数据
        - 已经是numpy数组
        
        Args:
            data: 输入数据
            model: 模型对象，用于判断是否需要转换数据格式
            
        Returns:
            转换后的numpy数组
        """
        import numpy as np
        from .data_source import AudioData
        from .feature import FeatureData
        
        if isinstance(data, np.ndarray):
            arr = data
        elif isinstance(data, list) and len(data) > 0:
            # 检查是否是AudioData列表
            if isinstance(data[0], AudioData):
                # 从AudioData中提取波形数据
                # AudioData.data 格式为 (channels, samples)
                waveforms = [audio.data for audio in data]
                arr = np.array(waveforms)  # (batch, channels, samples)
            # 检查是否是FeatureData列表
            elif isinstance(data[0], FeatureData):
                # 从FeatureData中提取特征数据
                # FeatureData.data 格式为 (channels, features, frames)
                features = [feat.data for feat in data]
                arr = np.array(features)  # (batch, channels, features, frames)
            else:
                # 假设是特征列表
                arr = np.array(data)
        else:
            arr = np.array(data)
        
        # 根据模型类型调整数据格式
        arr = self._adjust_data_format(arr, model)
        
        return arr
    
    def _adjust_data_format(self, data, model):
        """
        根据模型类型调整数据格式
        
        我们的数据默认为 NCHW (channels_first) 格式：
        - 2D 数据: (batch, channels, length)
        - 3D 数据: (batch, channels, height, width)
        
        Keras 模型默认使用 NHWC (channels_last) 格式：
        - 2D 数据: (batch, length, channels)
        - 3D 数据: (batch, height, width, channels)
        
        Args:
            data: numpy数组
            model: 模型对象
            
        Returns:
            调整格式后的numpy数组
        """
        import numpy as np
        
        if model is None:
            return data
        
        # 检查是否为 Keras 模型
        if self._is_keras_model(model):
            # Keras 模型需要 channels_last 格式
            data = self._convert_to_channels_last(data)
            logger.debug(f"数据已转换为 channels_last 格式: {data.shape}")
        # 如果未来支持 PyTorch 模型，保持 NCHW 格式不变
        
        return data
    
    def _is_keras_model(self, model) -> bool:
        """判断是否为 Keras 模型"""
        try:
            from tensorflow import keras
            return isinstance(model, keras.Model)
        except ImportError:
            return False
    
    def _convert_to_channels_last(self, data):
        """
        将数据从 channels_first (NCHW) 转换为 channels_last (NHWC)
        
        转换规则：
        - 3D 数据 (batch, channels, length) -> (batch, length, channels)
        - 4D 数据 (batch, channels, height, width) -> (batch, height, width, channels)
        
        判断是否需要转换的启发式规则：
        - 对于 3D 数据：如果 axis=1 的维度明显小于 axis=2，则认为是 channels_first
        - 对于 4D 数据：如果 axis=1 的维度明显小于 axis=2 和 axis=3，则认为是 channels_first
        """
        import numpy as np
        
        ndim = data.ndim
        
        if ndim == 3:
            # (batch, channels, length) -> (batch, length, channels)
            # 启发式判断：channels 通常很小（1-16），length 通常很大（>100）
            if data.shape[1] < data.shape[2]:
                return np.transpose(data, (0, 2, 1))
        
        elif ndim == 4:
            # (batch, channels, height, width) -> (batch, height, width, channels)
            # 启发式判断：channels 通常很小（1-16），height/width 通常较大
            if data.shape[1] < data.shape[2] and data.shape[1] < data.shape[3]:
                return np.transpose(data, (0, 2, 3, 1))
        
        # 1D 或 2D 数据不需要转换，或者已经是 channels_last 格式
        return data


@register_node
class ShowHistoryNode(BaseNode):
    """展示训练历史节点"""
    node_type = "show_history"
    display_name = "展示训练历史"
    category = NodeCategory.TRAINING
    description = "可视化展示训练历史曲线"
    icon = "📈"
    
    def _setup_ports(self):
        self.add_input("history", DataType.ANY, "训练历史")
        self.add_output("history", DataType.ANY, "训练历史(透传)")
    
    def _setup_parameters(self):
        self.add_parameter(
            "show_loss", "bool", True,
            display_name="显示损失曲线"
        )
        self.add_parameter(
            "show_accuracy", "bool", True,
            display_name="显示准确率曲线"
        )
        self.add_parameter(
            "save_path", "file", "",
            display_name="保存路径",
            description="可选，保存图表到文件",
            file_filter="PNG图像 (*.png);;JPEG图像 (*.jpg)"
        )
    
    def execute(self) -> bool:
        history = self.get_input_data("history")
        if history is None:
            self.error_message = "未提供训练历史"
            return False
        
        try:
            from src.visualization.metrics import MetricsPlotter
            import matplotlib.pyplot as plt
            
            # 确定要显示的指标
            metrics = []
            if self.get_parameter("show_loss"):
                metrics.append('loss')
            if self.get_parameter("show_accuracy"):
                metrics.append('accuracy')
            
            if not metrics:
                metrics = ['loss', 'accuracy']
            
            # 创建绘图器并绘制
            plotter = MetricsPlotter(figsize=(12, 5))
            fig = plotter.plot_training_history(history, metrics)
            
            # 保存图表（如果指定了路径）
            save_path = self.get_parameter("save_path")
            if save_path:
                plotter.save(save_path)
                logger.info(f"训练历史图表已保存到: {save_path}")
            
            # 显示图表
            plt.show()
            
            # 输出日志摘要
            epochs = len(history.get('loss', []))
            final_loss = history.get('loss', [0])[-1] if history.get('loss') else 'N/A'
            final_acc = history.get('accuracy', [0])[-1] if history.get('accuracy') else 'N/A'
            val_loss = history.get('val_loss', [0])[-1] if history.get('val_loss') else 'N/A'
            val_acc = history.get('val_accuracy', [0])[-1] if history.get('val_accuracy') else 'N/A'
            
            logger.info(f"训练历史摘要 ({epochs} epochs):")
            logger.info(f"  最终训练损失: {final_loss:.6f}" if isinstance(final_loss, float) else f"  最终训练损失: {final_loss}")
            logger.info(f"  最终训练准确率: {final_acc:.6f}" if isinstance(final_acc, float) else f"  最终训练准确率: {final_acc}")
            logger.info(f"  最终验证损失: {val_loss:.6f}" if isinstance(val_loss, float) else f"  最终验证损失: {val_loss}")
            logger.info(f"  最终验证准确率: {val_acc:.6f}" if isinstance(val_acc, float) else f"  最终验证准确率: {val_acc}")
            
            # 透传历史数据
            self.set_output_data("history", history)
            
            plotter.close()
            return True
            
        except Exception as e:
            self.error_message = f"展示训练历史失败: {str(e)}"
            logger.exception("展示训练历史异常")
            return False


@register_node
class ShowMetricsNode(BaseNode):
    """展示评估指标节点"""
    node_type = "show_metrics"
    display_name = "展示评估指标"
    category = NodeCategory.TRAINING
    description = "展示模型评估指标"
    icon = "📊"
    
    def _setup_ports(self):
        self.add_input("metrics", DataType.ANY, "评估指标")
        self.add_output("metrics", DataType.ANY, "评估指标(透传)")
    
    def _setup_parameters(self):
        self.add_parameter(
            "show_dialog", "bool", True,
            display_name="详细日志",
            description="是否输出详细指标日志（弹窗功能已移除）"
        )
    
    def execute(self) -> bool:
        metrics = self.get_input_data("metrics")
        if metrics is None:
            self.error_message = "未提供评估指标"
            return False
        
        try:
            # 打印指标到日志
            logger.info("=" * 50)
            logger.info("📊 模型评估结果:")
            logger.info("=" * 50)
            for name, value in metrics.items():
                if isinstance(value, float):
                    # 根据指标类型格式化显示
                    if 'accuracy' in name.lower() or 'acc' in name.lower():
                        logger.info(f"  {name}: {value:.4f} ({value*100:.2f}%)")
                    else:
                        logger.info(f"  {name}: {value:.6f}")
                else:
                    logger.info(f"  {name}: {value}")
            logger.info("=" * 50)
            
            # 注意：工作流执行在后台线程中，不能在后台线程中显示 GUI 对话框
            # 评估结果已通过日志输出，不再弹窗显示
            
            # 透传指标数据
            self.set_output_data("metrics", metrics)
            return True
            
        except Exception as e:
            self.error_message = f"展示评估指标失败: {str(e)}"
            logger.exception("展示评估指标异常")
            return False


@register_node
class PredictNode(BaseNode):
    """
    预测节点
    
    使用训练好的模型对输入数据进行预测推理。
    支持输出为标签、音频、一维特征或二维特征。
    """
    node_type = "predict"
    display_name = "预测"
    category = NodeCategory.TRAINING
    description = "使用训练好的模型进行预测"
    icon = "🔮"
    
    def _setup_ports(self):
        self.add_input("model", DataType.MODEL, "模型")
        self.add_input("input_data", DataType.ANY, "输入数据")
        self.add_output("output", DataType.ANY, "预测结果")
    
    def _setup_parameters(self):
        self.add_parameter(
            "output_type", "choice", "label",
            display_name="输出类型",
            choices=["label", "audio", "feature_1d", "feature_2d"],
            description="预测结果的数据类型：label=分类标签, audio=音频, feature_1d=一维特征, feature_2d=二维特征"
        )
        self.add_parameter(
            "batch_size", "int", 32,
            display_name="批次大小",
            min_value=1,
            description="预测时的批次大小"
        )
        self.add_parameter(
            "sample_rate", "int", 48000,
            display_name="采样率",
            min_value=8000,
            max_value=192000,
            description="输出音频的采样率（仅当输出类型为audio时使用）"
        )
    
    def execute(self) -> bool:
        try:
            import numpy as np
            from .data_source import AudioData
            from .feature import FeatureData
            
            model = self.get_input_data("model")
            input_data = self.get_input_data("input_data")
            
            if model is None:
                self.error_message = "未提供模型"
                return False
            
            if input_data is None:
                self.error_message = "未提供输入数据"
                return False
            
            # 转换输入数据为numpy数组
            X = self._convert_to_array(input_data, model)
            
            # 执行预测
            batch_size = self.get_parameter("batch_size")
            self.report_status(f"正在预测... (batch_size={batch_size})")
            
            predictions = model.predict(X, batch_size=batch_size, verbose=0)
            
            # 根据输出类型处理预测结果
            output_type = self.get_parameter("output_type")
            result = self._process_output(predictions, input_data, output_type)
            
            self.set_output_data("output", result)
            
            # 报告完成状态
            result_count = len(result) if isinstance(result, list) else 1
            self.report_status(f"预测完成: {result_count} 个样本, 输出类型={output_type}")
            logger.info(f"预测完成: {result_count} 个样本, 输出类型={output_type}")
            
            return True
            
        except Exception as e:
            self.error_message = f"预测失败: {str(e)}"
            logger.exception("预测异常")
            return False
    
    def _convert_to_array(self, data, model=None):
        """
        将输入数据转换为numpy数组，并根据模型类型调整数据格式
        
        支持:
        - 特征列表 (List[np.ndarray])
        - AudioData列表 (List[AudioData]) - 提取波形数据
        - FeatureData列表 (List[FeatureData]) - 提取特征数据
        - 已经是numpy数组
        """
        import numpy as np
        from .data_source import AudioData
        from .feature import FeatureData
        
        if isinstance(data, np.ndarray):
            arr = data
        elif isinstance(data, list) and len(data) > 0:
            # 检查是否是AudioData列表
            if isinstance(data[0], AudioData):
                waveforms = [audio.data for audio in data]
                arr = np.array(waveforms)  # (batch, channels, samples)
            # 检查是否是FeatureData列表
            elif isinstance(data[0], FeatureData):
                features = [feat.data for feat in data]
                arr = np.array(features)  # (batch, channels, features, frames)
            else:
                arr = np.array(data)
        else:
            arr = np.array(data)
        
        # 根据模型类型调整数据格式
        arr = self._adjust_data_format(arr, model)
        
        return arr
    
    def _adjust_data_format(self, data, model):
        """
        根据模型类型调整数据格式
        
        我们的数据默认为 NCHW (channels_first) 格式
        Keras 模型默认使用 NHWC (channels_last) 格式
        """
        import numpy as np
        
        if model is None:
            return data
        
        # 检查是否为 Keras 模型
        if self._is_keras_model(model):
            data = self._convert_to_channels_last(data)
            logger.debug(f"数据已转换为 channels_last 格式: {data.shape}")
        
        return data
    
    def _is_keras_model(self, model) -> bool:
        """判断是否为 Keras 模型"""
        try:
            from tensorflow import keras
            return isinstance(model, keras.Model)
        except ImportError:
            return False
    
    def _convert_to_channels_last(self, data):
        """
        将数据从 channels_first (NCHW) 转换为 channels_last (NHWC)
        """
        import numpy as np
        
        ndim = data.ndim
        
        if ndim == 3:
            # (batch, channels, length) -> (batch, length, channels)
            if data.shape[1] < data.shape[2]:
                return np.transpose(data, (0, 2, 1))
        
        elif ndim == 4:
            # (batch, channels, height, width) -> (batch, height, width, channels)
            if data.shape[1] < data.shape[2] and data.shape[1] < data.shape[3]:
                return np.transpose(data, (0, 2, 3, 1))
        
        return data
    
    def _convert_to_channels_first(self, data):
        """
        将数据从 channels_last (NHWC) 转换为 channels_first (NCHW)
        """
        import numpy as np
        
        ndim = data.ndim
        
        if ndim == 3:
            # (batch, length, channels) -> (batch, channels, length)
            if data.shape[2] < data.shape[1]:
                return np.transpose(data, (0, 2, 1))
        
        elif ndim == 4:
            # (batch, height, width, channels) -> (batch, channels, height, width)
            if data.shape[3] < data.shape[1] and data.shape[3] < data.shape[2]:
                return np.transpose(data, (0, 3, 1, 2))
        
        return data
    
    def _process_output(self, predictions, input_data, output_type: str):
        """
        根据输出类型处理预测结果
        
        Args:
            predictions: 模型预测输出 (numpy array)
            input_data: 原始输入数据（用于获取元信息）
            output_type: 输出类型 (label/audio/feature_1d/feature_2d)
        
        Returns:
            处理后的输出数据
        """
        import numpy as np
        from .data_source import AudioData
        from .feature import FeatureData
        
        if output_type == "label":
            # 分类标签：对预测概率取argmax
            if predictions.ndim > 1 and predictions.shape[-1] > 1:
                labels = np.argmax(predictions, axis=-1)
            else:
                # 二分类或回归，四舍五入
                labels = np.round(predictions.flatten()).astype(int)
            return labels.tolist()
        
        elif output_type == "audio":
            # 音频数据：重建 AudioData 列表
            # 先转换回 channels_first 格式
            predictions = self._convert_to_channels_first(predictions)
            
            sample_rate = self.get_parameter("sample_rate")
            audio_list = []
            
            for i in range(predictions.shape[0]):
                audio_data = predictions[i]  # (channels, samples)
                
                # 确保数据格式正确
                if audio_data.ndim == 1:
                    audio_data = audio_data.reshape(1, -1)
                
                # 获取源文件信息（如果有）
                source_file = ""
                if isinstance(input_data, list) and i < len(input_data):
                    if isinstance(input_data[i], AudioData):
                        source_file = input_data[i].file_path
                        sample_rate = input_data[i].sample_rate
                
                audio = AudioData(
                    data=audio_data.astype(np.float32),
                    sample_rate=sample_rate,
                    file_path=source_file,
                    duration=audio_data.shape[-1] / sample_rate
                )
                audio_list.append(audio)
            
            return audio_list
        
        elif output_type == "feature_1d":
            # 一维特征：重建 FeatureData 列表
            predictions = self._convert_to_channels_first(predictions)
            
            feature_list = []
            for i in range(predictions.shape[0]):
                feat_data = predictions[i]
                
                # 确保数据格式正确 (channels, features)
                if feat_data.ndim == 1:
                    feat_data = feat_data.reshape(1, -1)
                
                # 获取源文件信息
                source_file = ""
                sample_rate = 48000
                if isinstance(input_data, list) and i < len(input_data):
                    if hasattr(input_data[i], 'source_file'):
                        source_file = input_data[i].source_file
                    if hasattr(input_data[i], 'sample_rate'):
                        sample_rate = input_data[i].sample_rate
                
                feature = FeatureData(
                    data=feat_data.astype(np.float32),
                    feature_type="predicted_1d",
                    sample_rate=sample_rate,
                    hop_length=512,
                    source_file=source_file
                )
                feature_list.append(feature)
            
            return feature_list
        
        elif output_type == "feature_2d":
            # 二维特征：重建 FeatureData 列表
            predictions = self._convert_to_channels_first(predictions)
            
            feature_list = []
            for i in range(predictions.shape[0]):
                feat_data = predictions[i]
                
                # 确保数据格式正确 (channels, features, frames)
                if feat_data.ndim == 2:
                    feat_data = feat_data.reshape(1, *feat_data.shape)
                
                # 获取源文件信息
                source_file = ""
                sample_rate = 48000
                hop_length = 512
                if isinstance(input_data, list) and i < len(input_data):
                    if hasattr(input_data[i], 'source_file'):
                        source_file = input_data[i].source_file
                    if hasattr(input_data[i], 'sample_rate'):
                        sample_rate = input_data[i].sample_rate
                    if hasattr(input_data[i], 'hop_length'):
                        hop_length = input_data[i].hop_length
                
                feature = FeatureData(
                    data=feat_data.astype(np.float32),
                    feature_type="predicted_2d",
                    sample_rate=sample_rate,
                    hop_length=hop_length,
                    source_file=source_file
                )
                feature_list.append(feature)
            
            return feature_list
        
        else:
            # 默认返回原始预测结果
            return predictions
