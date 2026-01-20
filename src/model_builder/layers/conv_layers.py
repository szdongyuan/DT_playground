# -*- coding: utf-8 -*-
"""
卷积层节点
"""

from ..layer_base import LayerCategory, LayerNode, register_layer


@register_layer
class Conv1DLayer(LayerNode):
    """1D卷积层"""
    layer_type = "conv1d"
    display_name = "Conv1D 一维卷积"
    category = LayerCategory.CONV
    description = "一维卷积层，用于时序数据"
    icon = "📈"
    keras_class = "Conv1D"
    
    def _setup_parameters(self):
        self.add_parameter(
            "filters", "int", 32,
            display_name="卷积核数量",
            description="输出空间的维度（卷积核的数量）",
            min_value=1
        )
        self.add_parameter(
            "kernel_size", "int", 3,
            display_name="卷积核大小",
            description="卷积窗口的长度",
            min_value=1
        )
        self.add_parameter(
            "strides", "int", 1,
            display_name="步长",
            description="卷积的步长",
            min_value=1
        )
        self.add_parameter(
            "padding", "choice", "same",
            display_name="填充方式",
            choices=["valid", "same", "causal"],
            description="填充模式"
        )
        self.add_parameter(
            "activation", "choice", "relu",
            display_name="激活函数",
            choices=["None", "relu", "sigmoid", "tanh", "linear", "leaky_relu", "elu"],
            description="激活函数类型"
        )
        self.add_parameter(
            "use_bias", "bool", True,
            display_name="使用偏置",
            description="是否使用偏置向量"
        )
    
    def build_keras_layer(self):
        from tensorflow import keras
        
        activation = self.get_parameter("activation")
        if activation == "None":
            activation = None
        
        return keras.layers.Conv1D(
            filters=self.get_parameter("filters"),
            kernel_size=self.get_parameter("kernel_size"),
            strides=self.get_parameter("strides"),
            padding=self.get_parameter("padding"),
            activation=activation,
            use_bias=self.get_parameter("use_bias"),
            name=self.layer_name or None
        )


@register_layer
class Conv2DLayer(LayerNode):
    """2D卷积层"""
    layer_type = "conv2d"
    display_name = "Conv2D 二维卷积"
    category = LayerCategory.CONV
    description = "二维卷积层，用于图像或频谱数据"
    icon = "🖼️"
    keras_class = "Conv2D"
    
    def _setup_parameters(self):
        self.add_parameter(
            "filters", "int", 32,
            display_name="卷积核数量",
            description="输出空间的维度（卷积核的数量）",
            min_value=1
        )
        self.add_parameter(
            "kernel_size", "str", "(3, 3)",
            display_name="卷积核大小",
            description="卷积窗口的高度和宽度"
        )
        self.add_parameter(
            "strides", "str", "(1, 1)",
            display_name="步长",
            description="卷积的步长"
        )
        self.add_parameter(
            "padding", "choice", "same",
            display_name="填充方式",
            choices=["valid", "same"],
            description="填充模式"
        )
        self.add_parameter(
            "activation", "choice", "relu",
            display_name="激活函数",
            choices=["None", "relu", "sigmoid", "tanh", "linear", "leaky_relu", "elu"],
            description="激活函数类型"
        )
        self.add_parameter(
            "use_bias", "bool", True,
            display_name="使用偏置",
            description="是否使用偏置向量"
        )
    
    def build_keras_layer(self):
        from tensorflow import keras
        
        kernel_size = eval(self.get_parameter("kernel_size"))
        strides = eval(self.get_parameter("strides"))
        activation = self.get_parameter("activation")
        if activation == "None":
            activation = None
        
        return keras.layers.Conv2D(
            filters=self.get_parameter("filters"),
            kernel_size=kernel_size,
            strides=strides,
            padding=self.get_parameter("padding"),
            activation=activation,
            use_bias=self.get_parameter("use_bias"),
            name=self.layer_name or None
        )


@register_layer
class SeparableConv1DLayer(LayerNode):
    """深度可分离1D卷积层"""
    layer_type = "separable_conv1d"
    display_name = "SeparableConv1D"
    category = LayerCategory.CONV
    description = "深度可分离一维卷积"
    icon = "📊"
    keras_class = "SeparableConv1D"
    
    def _setup_parameters(self):
        self.add_parameter(
            "filters", "int", 32,
            display_name="卷积核数量",
            min_value=1
        )
        self.add_parameter(
            "kernel_size", "int", 3,
            display_name="卷积核大小",
            min_value=1
        )
        self.add_parameter(
            "strides", "int", 1,
            display_name="步长",
            min_value=1
        )
        self.add_parameter(
            "padding", "choice", "same",
            display_name="填充方式",
            choices=["valid", "same"]
        )
        self.add_parameter(
            "activation", "choice", "relu",
            display_name="激活函数",
            choices=["None", "relu", "sigmoid", "tanh", "linear", "leaky_relu", "elu"]
        )
    
    def build_keras_layer(self):
        from tensorflow import keras
        
        activation = self.get_parameter("activation")
        if activation == "None":
            activation = None
        
        return keras.layers.SeparableConv1D(
            filters=self.get_parameter("filters"),
            kernel_size=self.get_parameter("kernel_size"),
            strides=self.get_parameter("strides"),
            padding=self.get_parameter("padding"),
            activation=activation,
            name=self.layer_name or None
        )


@register_layer
class Conv1DTransposeLayer(LayerNode):
    """1D转置卷积层（反卷积）"""
    layer_type = "conv1d_transpose"
    display_name = "Conv1DTranspose 转置卷积"
    category = LayerCategory.CONV
    description = "一维转置卷积层，用于上采样"
    icon = "⬆️"
    keras_class = "Conv1DTranspose"
    
    def _setup_parameters(self):
        self.add_parameter(
            "filters", "int", 32,
            display_name="卷积核数量",
            description="输出空间的维度（卷积核的数量）",
            min_value=1
        )
        self.add_parameter(
            "kernel_size", "int", 3,
            display_name="卷积核大小",
            description="卷积窗口的长度",
            min_value=1
        )
        self.add_parameter(
            "strides", "int", 2,
            display_name="步长",
            description="卷积的步长，控制上采样倍数",
            min_value=1
        )
        self.add_parameter(
            "padding", "choice", "same",
            display_name="填充方式",
            choices=["valid", "same"],
            description="填充模式"
        )
        self.add_parameter(
            "activation", "choice", "relu",
            display_name="激活函数",
            choices=["None", "relu", "sigmoid", "tanh", "linear", "leaky_relu", "elu"],
            description="激活函数类型"
        )
        self.add_parameter(
            "use_bias", "bool", True,
            display_name="使用偏置",
            description="是否使用偏置向量"
        )
    
    def build_keras_layer(self):
        from tensorflow import keras
        
        activation = self.get_parameter("activation")
        if activation == "None":
            activation = None
        
        return keras.layers.Conv1DTranspose(
            filters=self.get_parameter("filters"),
            kernel_size=self.get_parameter("kernel_size"),
            strides=self.get_parameter("strides"),
            padding=self.get_parameter("padding"),
            activation=activation,
            use_bias=self.get_parameter("use_bias"),
            name=self.layer_name or None
        )


@register_layer
class Conv2DTransposeLayer(LayerNode):
    """2D转置卷积层（反卷积）"""
    layer_type = "conv2d_transpose"
    display_name = "Conv2DTranspose 转置卷积"
    category = LayerCategory.CONV
    description = "二维转置卷积层，用于上采样"
    icon = "⬆️"
    keras_class = "Conv2DTranspose"
    
    def _setup_parameters(self):
        self.add_parameter(
            "filters", "int", 32,
            display_name="卷积核数量",
            description="输出空间的维度（卷积核的数量）",
            min_value=1
        )
        self.add_parameter(
            "kernel_size", "str", "(3, 3)",
            display_name="卷积核大小",
            description="卷积窗口的高度和宽度"
        )
        self.add_parameter(
            "strides", "str", "(2, 2)",
            display_name="步长",
            description="卷积的步长，控制上采样倍数"
        )
        self.add_parameter(
            "padding", "choice", "same",
            display_name="填充方式",
            choices=["valid", "same"],
            description="填充模式"
        )
        self.add_parameter(
            "activation", "choice", "relu",
            display_name="激活函数",
            choices=["None", "relu", "sigmoid", "tanh", "linear", "leaky_relu", "elu"],
            description="激活函数类型"
        )
        self.add_parameter(
            "use_bias", "bool", True,
            display_name="使用偏置",
            description="是否使用偏置向量"
        )
    
    def build_keras_layer(self):
        from tensorflow import keras
        
        kernel_size = eval(self.get_parameter("kernel_size"))
        strides = eval(self.get_parameter("strides"))
        activation = self.get_parameter("activation")
        if activation == "None":
            activation = None
        
        return keras.layers.Conv2DTranspose(
            filters=self.get_parameter("filters"),
            kernel_size=kernel_size,
            strides=strides,
            padding=self.get_parameter("padding"),
            activation=activation,
            use_bias=self.get_parameter("use_bias"),
            name=self.layer_name or None
        )
