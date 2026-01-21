# -*- coding: utf-8 -*-
"""
Input/Output Layer Nodes
"""

from typing import Tuple

from ..layer_base import LayerCategory, LayerNode, register_layer
from ..model_graph import CompileConfig


@register_layer
class InputLayer(LayerNode):
    """输入层"""
    layer_type = "input"
    display_name = "Input 输入层"
    category = LayerCategory.INPUT
    description = "定义模型输入的形状"
    icon = "📥"
    keras_class = "Input"
    
    def _setup_parameters(self):
        self.add_parameter(
            "shape", "str", "(128, 1)",
            display_name="输入形状",
            description="输入数据的形状，例如 (128,) 或 (128, 64) 或 (32, 32, 3)"
        )
        self.add_parameter(
            "dtype", "choice", "float32",
            display_name="数据类型",
            choices=["float32", "float64", "int32", "int64"],
            description="输入数据的数据类型"
        )
        self.add_parameter(
            "name", "str", "",
            display_name="层名称",
            description="可选的层名称",
            required=False
        )
    
    def validate(self) -> Tuple[bool, str]:
        """验证输入层配置"""
        shape_str = self.get_parameter("shape")
        
        # 检查 shape 是否为空
        if not shape_str or not shape_str.strip():
            return False, "输入形状不能为空"
        
        # 尝试解析 shape
        try:
            shape = eval(shape_str)
            if not isinstance(shape, tuple):
                return False, f"输入形状必须是元组，当前为: {type(shape).__name__}"
            if len(shape) == 0:
                return False, "输入形状元组不能为空"
            for dim in shape:
                if dim is not None and (not isinstance(dim, int) or dim <= 0):
                    return False, f"输入形状维度必须是正整数或None，当前为: {dim}"
        except Exception as e:
            return False, f"输入形状格式错误: {e}"
        
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
    display_name = "Output 输出层"
    category = LayerCategory.OUTPUT
    description = "模型输出层，包含编译配置（优化器、损失函数、指标）"
    icon = "📤"
    keras_class = "Identity"
    
    def _setup_parameters(self):
        # 激活函数（输出层特有）
        self.add_parameter(
            "activation", "choice", "none",
            display_name="激活函数",
            description="输出层的激活函数（none 表示不使用激活函数）",
            choices=["none", "linear", "sigmoid", "softmax", "tanh", "relu"]
        )
        
        # 编译配置 - 优化器（使用 CompileConfig 的常量）
        self.add_parameter(
            "optimizer", "choice", "Adam",
            display_name="优化器",
            description="训练优化器",
            choices=CompileConfig.OPTIMIZERS
        )
        
        # 学习率
        self.add_parameter(
            "learning_rate", "float", 0.001,
            display_name="学习率",
            description="优化器学习率",
            min_value=0.000001,
            max_value=1.0
        )
        
        # 损失函数（使用 CompileConfig 的常量）
        self.add_parameter(
            "loss", "choice", "mse",
            display_name="损失函数",
            description="训练损失函数",
            choices=CompileConfig.LOSSES
        )
        
        # 评估指标（多选，使用 CompileConfig 的常量）
        self.add_parameter(
            "metrics", "multichoice", ["mae"],
            display_name="评估指标",
            description="选择评估指标（可多选）",
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

