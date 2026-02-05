# -*- coding: utf-8 -*-
"""
Convolutional Layer Nodes
"""

from ..layer_base import LayerCategory, LayerNode, register_layer
from src.ui.i18n import tr_
@register_layer
class Conv1DLayer(LayerNode):
    """1D卷积层"""
    layer_type = "conv1d"
    display_name = tr_("Conv1D")
    category = LayerCategory.CONV
    description = tr_("1D convolution layer for sequence/time-series data")
    icon = "📈"
    keras_class = "Conv1D"
    
    def _setup_parameters(self):
        self.add_parameter(
            "filters", "int", 32,
            display_name=tr_("Number of filters"),
            description=tr_("Dimensionality of the output space (number of convolution filters)"),
            min_value=1
        )
        self.add_parameter(
            "kernel_size", "int", 3,
            display_name=tr_("Kernel size"),
            description=tr_("Length of the 1D convolution window"),
            min_value=1
        )
        self.add_parameter(
            "strides", "int", 1,
            display_name=tr_("Stride"),
            description=tr_("Stride length of the convolution"),
            min_value=1
        )
        self.add_parameter(
            "padding", "choice", "same",
            display_name=tr_("Padding"),
            choices=["valid", "same", "causal"],
            description=tr_("Padding mode")
        )
        self.add_parameter(
            "activation", "choice", "relu",
            display_name=tr_("Activation"),
            choices=["None", "relu", "sigmoid", "tanh", "linear", "leaky_relu", "elu"],
            description=tr_("Activation function")
        )
        self.add_parameter(
            "use_bias", "bool", True,
            display_name=tr_("Use bias"),
            description=tr_("Whether the layer uses a bias vector")
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
    display_name = tr_("Conv2D")
    category = LayerCategory.CONV
    description = tr_("2D convolution layer for images or spectrograms")
    icon = "🖼️"
    keras_class = "Conv2D"
    
    def _setup_parameters(self):
        self.add_parameter(
            "filters", "int", 32,
            display_name=tr_("Number of filters"),
            description=tr_("Dimensionality of the output space (number of convolution filters)"),
            min_value=1
        )
        self.add_parameter(
            "kernel_size", "str", "(3, 3)",
            display_name=tr_("Kernel size"),
            description=tr_("Height and width of the 2D convolution window")
        )
        self.add_parameter(
            "strides", "str", "(1, 1)",
            display_name=tr_("Stride"),
            description=tr_("Stride of the convolution")
        )
        self.add_parameter(
            "padding", "choice", "same",
            display_name=tr_("Padding"),
            choices=["valid", "same"],
            description=tr_("Padding mode")
        )
        self.add_parameter(
            "activation", "choice", "relu",
            display_name=tr_("Activation"),
            choices=["None", "relu", "sigmoid", "tanh", "linear", "leaky_relu", "elu"],
            description=tr_("Activation function")
        )
        self.add_parameter(
            "use_bias", "bool", True,
            display_name=tr_("Use bias"),
            description=tr_("Whether the layer uses a bias vector")
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
    description = tr_("Depthwise separable 1D convolution")
    icon = "📊"
    keras_class = "SeparableConv1D"
    
    def _setup_parameters(self):
        self.add_parameter(
            "filters", "int", 32,
            display_name=tr_("Number of filters"),
            min_value=1
        )
        self.add_parameter(
            "kernel_size", "int", 3,
            display_name=tr_("Kernel size"),
            min_value=1
        )
        self.add_parameter(
            "strides", "int", 1,
            display_name=tr_("Stride"),
            min_value=1
        )
        self.add_parameter(
            "padding", "choice", "same",
            display_name=tr_("Padding"),
            choices=["valid", "same"]
        )
        self.add_parameter(
            "activation", "choice", "relu",
            display_name=tr_("Activation"),
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
    display_name = tr_("Conv1DTranspose")
    category = LayerCategory.CONV
    description = tr_("1D transposed convolution layer for upsampling")
    icon = "⬆️"
    keras_class = "Conv1DTranspose"
    
    def _setup_parameters(self):
        self.add_parameter(
            "filters", "int", 32,
            display_name=tr_("Number of filters"),
            description=tr_("Dimensionality of the output space (number of convolution filters)"),
            min_value=1
        )
        self.add_parameter(
            "kernel_size", "int", 3,
            display_name=tr_("Kernel size"),
            description=tr_("Length of the 1D convolution window"),
            min_value=1
        )
        self.add_parameter(
            "strides", "int", 2,
            display_name=tr_("Stride"),
            description=tr_("Stride length of the convolution (controls upsampling factor)"),
            min_value=1
        )
        self.add_parameter(
            "padding", "choice", "same",
            display_name=tr_("Padding"),
            choices=["valid", "same"],
            description=tr_("Padding mode")
        )
        self.add_parameter(
            "activation", "choice", "relu",
            display_name=tr_("Activation"),
            choices=["None", "relu", "sigmoid", "tanh", "linear", "leaky_relu", "elu"],
            description=tr_("Activation function")
        )
        self.add_parameter(
            "use_bias", "bool", True,
            display_name=tr_("Use bias"),
            description=tr_("Whether the layer uses a bias vector")
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
    display_name = tr_("Conv2DTranspose")
    category = LayerCategory.CONV
    description = tr_("2D transposed convolution layer for upsampling")
    icon = "⬆️"
    keras_class = "Conv2DTranspose"
    
    def _setup_parameters(self):
        self.add_parameter(
            "filters", "int", 32,
            display_name=tr_("Number of filters"),
            description=tr_("Dimensionality of the output space (number of convolution filters)"),
            min_value=1
        )
        self.add_parameter(
            "kernel_size", "str", "(3, 3)",
            display_name=tr_("Kernel size"),
            description=tr_("Height and width of the 2D convolution window")
        )
        self.add_parameter(
            "strides", "str", "(2, 2)",
            display_name=tr_("Stride"),
            description=tr_("Stride of the convolution (controls upsampling factor)")
        )
        self.add_parameter(
            "padding", "choice", "same",
            display_name=tr_("Padding"),
            choices=["valid", "same"],
            description=tr_("Padding mode")
        )
        self.add_parameter(
            "activation", "choice", "relu",
            display_name=tr_("Activation"),
            choices=["None", "relu", "sigmoid", "tanh", "linear", "leaky_relu", "elu"],
            description=tr_("Activation function")
        )
        self.add_parameter(
            "use_bias", "bool", True,
            display_name=tr_("Use bias"),
            description=tr_("Whether the layer uses a bias vector")
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
