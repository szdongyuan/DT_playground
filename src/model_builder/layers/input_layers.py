# -*- coding: utf-8 -*-
"""
Input/Output Layer Nodes
"""

from typing import Tuple

from ..layer_base import LayerCategory, LayerNode, register_layer
from ..model_graph import CompileConfig
from src.ui.i18n import tr_
@register_layer
class InputLayer(LayerNode):
    """输入层"""
    layer_type = "input"
    display_name = tr_("Input")
    category = LayerCategory.INPUT
    description = tr_("Define the model input shape")
    icon = "📥"
    keras_class = "Input"
    
    def _setup_parameters(self):
        self.add_parameter(
            "shape", "str", "(128, 1)",
            display_name=tr_("Input shape"),
            description=tr_("Input data shape, e.g. (128,) or (128, 64) or (32, 32, 3)")
        )
        self.add_parameter(
            "dtype", "choice", "float32",
            display_name=tr_("Data type"),
            choices=["float32", "float64", "int32", "int64"],
            description=tr_("Data type of the input")
        )
        self.add_parameter(
            "name", "str", "",
            display_name=tr_("Layer name"),
            description=tr_("Optional layer name"),
            required=False
        )
    
    def validate(self) -> Tuple[bool, str]:
        """验证输入层配置"""
        shape_str = self.get_parameter("shape")
        
        # 检查 shape 是否为空
        if not shape_str or not shape_str.strip():
            return False, tr_("Input shape cannot be empty")
        
        # 尝试解析 shape
        try:
            shape = eval(shape_str)
            if not isinstance(shape, tuple):
                return False, tr_("Input shape must be a tuple, got: {type_name}").format(
                    type_name=type(shape).__name__
                )
            if len(shape) == 0:
                return False, tr_("Input shape tuple cannot be empty")
            for dim in shape:
                if dim is not None and (not isinstance(dim, int) or dim <= 0):
                    return False, tr_("Input shape dimensions must be positive integers or None, got: {dim}").format(
                        dim=dim
                    )
        except Exception as e:
            return False, tr_("Invalid input shape format: {error}").format(error=str(e))
        
        return True, ""
    
    def build_keras_layer(self):
        from tensorflow import keras
        
        # 解析形状字符串
        shape_str = self.get_parameter("shape")
        shape = eval(shape_str)  # 将字符串转换为元组
        
        kwargs = {
            "shape": shape,
            "dtype": self.get_parameter("dtype"),
        }
        
        name = self.get_parameter("name")
        if name:
            kwargs["name"] = name
        
        return keras.layers.Input(**kwargs)
    
    def compute_output_shape(self, input_shape):
        shape_str = self.get_parameter("shape")
        return eval(shape_str)


@register_layer
class OutputLayer(LayerNode):
    """
    输出层
    
    标记模型输出并配置编译选项（优化器、损失函数、评估指标）。
    编译配置常量统一引用自 CompileConfig。
    """
    layer_type = "output"
    display_name = tr_("Output")
    category = LayerCategory.OUTPUT
    description = tr_("Model output layer with compile settings (optimizer, loss, metrics)")
    icon = "📤"
    keras_class = "Identity"
    
    def _setup_parameters(self):
        # 激活函数（输出层特有）
        self.add_parameter(
            "activation", "choice", "none",
            display_name=tr_("Activation"),
            description=tr_("Activation function for output layer ('none' means no activation)"),
            choices=["none", "linear", "sigmoid", "softmax", "tanh", "relu"]
        )
        
        # 编译配置 - 优化器（使用 CompileConfig 的常量）
        self.add_parameter(
            "optimizer", "choice", "Adam",
            display_name=tr_("Optimizer"),
            description=tr_("Training optimizer"),
            choices=CompileConfig.OPTIMIZERS
        )
        
        # 学习率
        self.add_parameter(
            "learning_rate", "float", 0.001,
            display_name=tr_("Learning rate"),
            description=tr_("Optimizer learning rate"),
            min_value=0.000001,
            max_value=1.0
        )
        
        # 损失函数（使用 CompileConfig 的常量）
        self.add_parameter(
            "loss", "choice", "mse",
            display_name=tr_("Loss"),
            description=tr_("Training loss function"),
            choices=CompileConfig.LOSSES
        )
        
        # 评估指标（多选，使用 CompileConfig 的常量）
        self.add_parameter(
            "metrics", "multichoice", ["mae"],
            display_name=tr_("Metrics"),
            description=tr_("Select metrics (multi-select)"),
            choices=CompileConfig.METRICS
        )
    
    def get_compile_config(self) -> dict:
        """获取编译配置"""
        metrics = self.get_parameter("metrics")
        # 确保 metrics 是列表
        if isinstance(metrics, str):
            metrics = [m.strip() for m in metrics.split(",") if m.strip()]
        if not metrics:
            metrics = ["mae"]
        
        return {
            "optimizer": self.get_parameter("optimizer"),
            "learning_rate": self.get_parameter("learning_rate"),
            "loss": self.get_parameter("loss"),
            "metrics": metrics,
        }
    
    def build_keras_layer(self):
        """
        构建输出层
        
        使用 Activation 层，none/linear 激活函数相当于恒等映射
        """
        from tensorflow import keras
        
        activation = self.get_parameter("activation")
        
        # "none" 表示不使用激活函数，使用 linear 作为恒等映射
        if activation == "none":
            activation = "linear"
        
        # 始终使用 Activation 层（避免 Lambda 层的序列化问题）
        # linear 激活函数相当于恒等映射 f(x) = x
        return keras.layers.Activation(
            activation, 
            name=self.layer_name or "output"
        )

