# -*- coding: utf-8 -*-
"""
Core Layer Nodes (Dense, Embedding, etc.)
"""

from ..layer_base import LayerCategory, LayerNode, register_layer


@register_layer
class DenseLayer(LayerNode):
    """全连接层"""
    layer_type = "dense"
    display_name = "Dense 全连接层"
    category = LayerCategory.CORE
    description = "标准全连接神经网络层"
    icon = "🔗"
    keras_class = "Dense"
    
    def _setup_parameters(self):
        self.add_parameter(
            "units", "int", 64,
            display_name="神经元数量",
            description="输出空间的维度",
            min_value=1
        )
        self.add_parameter(
            "activation", "choice", "relu",
            display_name="激活函数",
            choices=["none", "relu", "sigmoid", "tanh", "softmax", "linear", "leaky_relu", "elu", "selu", "swish", "gelu"],
            description="激活函数类型（none 表示不使用激活函数）"
        )
        self.add_parameter(
            "use_bias", "bool", True,
            display_name="使用偏置",
            description="是否添加偏置向量"
        )
        self.add_parameter(
            "kernel_initializer", "choice", "glorot_uniform",
            display_name="权重初始化",
            choices=["glorot_uniform", "glorot_normal", "he_uniform", "he_normal", "zeros", "ones", "random_normal"],
            description="权重矩阵的初始化方法"
        )
    
    def build_keras_layer(self):
        from tensorflow import keras
        
        activation = self.get_parameter("activation")
        # "none" 表示不使用激活函数
        if activation == "none":
            activation = None
        
        return keras.layers.Dense(
            units=self.get_parameter("units"),
            activation=activation,
            use_bias=self.get_parameter("use_bias"),
            kernel_initializer=self.get_parameter("kernel_initializer"),
            name=self.layer_name or None
        )


@register_layer
class EmbeddingLayer(LayerNode):
    """嵌入层"""
    layer_type = "embedding"
    display_name = "Embedding 嵌入层"
    category = LayerCategory.CORE
    description = "将正整数索引转换为固定大小的密集向量"
    icon = "📊"
    keras_class = "Embedding"
    
    def _setup_parameters(self):
        self.add_parameter(
            "input_dim", "int", 10000,
            display_name="词汇量大小",
            description="词汇表的大小",
            min_value=1
        )
        self.add_parameter(
            "output_dim", "int", 128,
            display_name="嵌入维度",
            description="密集嵌入的维度",
            min_value=1
        )
        self.add_parameter(
            "mask_zero", "bool", False,
            display_name="掩码零值",
            description="是否将输入值0作为需要被屏蔽的特殊填充值"
        )
    
    def build_keras_layer(self):
        from tensorflow import keras
        
        return keras.layers.Embedding(
            input_dim=self.get_parameter("input_dim"),
            output_dim=self.get_parameter("output_dim"),
            mask_zero=self.get_parameter("mask_zero"),
            name=self.layer_name or None
        )

