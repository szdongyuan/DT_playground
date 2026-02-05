# -*- coding: utf-8 -*-
"""
Core Layer Nodes (Dense, Embedding, etc.)
"""

from ..layer_base import LayerCategory, LayerNode, register_layer
from src.ui.i18n import tr_
@register_layer
class DenseLayer(LayerNode):
    """全连接层"""
    layer_type = "dense"
    display_name = tr_("Dense")
    category = LayerCategory.CORE
    description = tr_("Standard fully-connected neural network layer")
    icon = "🔗"
    keras_class = "Dense"
    
    def _setup_parameters(self):
        self.add_parameter(
            "units", "int", 64,
            display_name=tr_("Units"),
            description=tr_("Dimensionality of the output space"),
            min_value=1
        )
        self.add_parameter(
            "activation", "choice", "relu",
            display_name=tr_("Activation"),
            choices=["none", "relu", "sigmoid", "tanh", "softmax", "linear", "leaky_relu", "elu", "selu", "swish", "gelu"],
            description=tr_("Activation function ('none' means no activation)")
        )
        self.add_parameter(
            "use_bias", "bool", True,
            display_name=tr_("Use bias"),
            description=tr_("Whether the layer uses a bias vector")
        )
        self.add_parameter(
            "kernel_initializer", "choice", "glorot_uniform",
            display_name=tr_("Kernel initializer"),
            choices=["glorot_uniform", "glorot_normal", "he_uniform", "he_normal", "zeros", "ones", "random_normal"],
            description=tr_("Initializer for the kernel weights matrix")
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
    display_name = tr_("Embedding")
    category = LayerCategory.CORE
    description = tr_("Turns positive integers (indexes) into dense vectors of fixed size")
    icon = "📊"
    keras_class = "Embedding"
    
    def _setup_parameters(self):
        self.add_parameter(
            "input_dim", "int", 10000,
            display_name=tr_("Vocabulary size"),
            description=tr_("Size of the vocabulary"),
            min_value=1
        )
        self.add_parameter(
            "output_dim", "int", 128,
            display_name=tr_("Embedding dimension"),
            description=tr_("Dimension of the dense embedding"),
            min_value=1
        )
        self.add_parameter(
            "mask_zero", "bool", False,
            display_name=tr_("Mask zero"),
            description=tr_("Whether to treat input value 0 as a special padding value to be masked")
        )
    
    def build_keras_layer(self):
        from tensorflow import keras
        
        return keras.layers.Embedding(
            input_dim=self.get_parameter("input_dim"),
            output_dim=self.get_parameter("output_dim"),
            mask_zero=self.get_parameter("mask_zero"),
            name=self.layer_name or None
        )

