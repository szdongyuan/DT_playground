# -*- coding: utf-8 -*-
"""
Activation Function Layer Nodes
"""

from ..layer_base import LayerCategory, LayerNode, register_layer


@register_layer
class ActivationLayer(LayerNode):
    """激活函数层"""
    layer_type = "activation"
    display_name = "Activation 激活函数"
    category = LayerCategory.ACTIVATION
    description = "应用激活函数"
    icon = "⚡"
    keras_class = "Activation"
    
    def _setup_parameters(self):
        self.add_parameter(
            "activation", "choice", "relu",
            display_name="激活函数",
            choices=[
                "relu", "sigmoid", "tanh", "softmax", "softplus",
                "softsign", "elu", "selu", "exponential", "linear",
                "leaky_relu", "gelu", "swish", "mish"
            ]
        )
    
    def build_keras_layer(self):
        from tensorflow import keras
        
        return keras.layers.Activation(
            self.get_parameter("activation"),
            name=self.layer_name or None
        )


@register_layer
class LeakyReLULayer(LayerNode):
    """LeakyReLU层"""
    layer_type = "leaky_relu"
    display_name = "LeakyReLU"
    category = LayerCategory.ACTIVATION
    description = "带泄漏的ReLU激活函数"
    icon = "📈"
    keras_class = "LeakyReLU"
    
    def _setup_parameters(self):
        self.add_parameter(
            "alpha", "float", 0.3,
            display_name="Alpha",
            description="负斜率系数",
            min_value=0.0
        )
    
    def build_keras_layer(self):
        from tensorflow import keras
        
        return keras.layers.LeakyReLU(
            alpha=self.get_parameter("alpha"),
            name=self.layer_name or None
        )


@register_layer
class PReLULayer(LayerNode):
    """PReLU层"""
    layer_type = "prelu"
    display_name = "PReLU"
    category = LayerCategory.ACTIVATION
    description = "参数化ReLU激活函数"
    icon = "📈"
    keras_class = "PReLU"
    
    def _setup_parameters(self):
        pass  # 参数会自动学习
    
    def build_keras_layer(self):
        from tensorflow import keras
        return keras.layers.PReLU(name=self.layer_name or None)


@register_layer
class ELULayer(LayerNode):
    """ELU层"""
    layer_type = "elu"
    display_name = "ELU"
    category = LayerCategory.ACTIVATION
    description = "指数线性单元"
    icon = "📊"
    keras_class = "ELU"
    
    def _setup_parameters(self):
        self.add_parameter(
            "alpha", "float", 1.0,
            display_name="Alpha",
            min_value=0.0
        )
    
    def build_keras_layer(self):
        from tensorflow import keras
        
        return keras.layers.ELU(
            alpha=self.get_parameter("alpha"),
            name=self.layer_name or None
        )


@register_layer
class SoftmaxLayer(LayerNode):
    """Softmax层"""
    layer_type = "softmax"
    display_name = "Softmax"
    category = LayerCategory.ACTIVATION
    description = "Softmax激活函数"
    icon = "📊"
    keras_class = "Softmax"
    
    def _setup_parameters(self):
        self.add_parameter(
            "axis", "int", -1,
            display_name="轴",
            description="应用softmax的轴"
        )
    
    def build_keras_layer(self):
        from tensorflow import keras
        
        return keras.layers.Softmax(
            axis=self.get_parameter("axis"),
            name=self.layer_name or None
        )

