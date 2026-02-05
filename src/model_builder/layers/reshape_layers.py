# -*- coding: utf-8 -*-
"""
Shape Transformation Layer Nodes
"""

from ..layer_base import LayerCategory, LayerNode, register_layer
from src.ui.i18n import tr_
@register_layer
class FlattenLayer(LayerNode):
    """展平层"""
    layer_type = "flatten"
    display_name = tr_("Flatten")
    category = LayerCategory.RESHAPE
    description = tr_("Flattens the input to 1D")
    icon = "📋"
    keras_class = "Flatten"
    
    def _setup_parameters(self):
        pass  # 无参数
    
    def build_keras_layer(self):
        from tensorflow import keras
        return keras.layers.Flatten(name=self.layer_name or None)


@register_layer
class ReshapeLayer(LayerNode):
    """重塑层"""
    layer_type = "reshape"
    display_name = tr_("Reshape")
    category = LayerCategory.RESHAPE
    description = tr_("Reshapes the input to the given shape")
    icon = "🔄"
    keras_class = "Reshape"
    
    def _setup_parameters(self):
        self.add_parameter(
            "target_shape", "str", "(64, 1)",
            display_name=tr_("Target shape"),
            description=tr_("Target shape (excluding batch dimension)")
        )
    
    def build_keras_layer(self):
        from tensorflow import keras
        
        target_shape = eval(self.get_parameter("target_shape"))
        return keras.layers.Reshape(
            target_shape=target_shape,
            name=self.layer_name or None
        )


@register_layer
class PermuteLayer(LayerNode):
    """置换层"""
    layer_type = "permute"
    display_name = tr_("Permute")
    category = LayerCategory.RESHAPE
    description = tr_("Permutes the dimensions of the input")
    icon = "🔀"
    keras_class = "Permute"
    
    def _setup_parameters(self):
        self.add_parameter(
            "dims", "str", "(2, 1)",
            display_name=tr_("Dimension order"),
            description=tr_("Permutation pattern (1-based indices)")
        )
    
    def build_keras_layer(self):
        from tensorflow import keras
        
        dims = eval(self.get_parameter("dims"))
        return keras.layers.Permute(
            dims=dims,
            name=self.layer_name or None
        )


@register_layer
class RepeatVectorLayer(LayerNode):
    """重复向量层"""
    layer_type = "repeat_vector"
    display_name = "RepeatVector"
    category = LayerCategory.RESHAPE
    description = tr_("Repeats the input n times")
    icon = "🔁"
    keras_class = "RepeatVector"
    
    def _setup_parameters(self):
        self.add_parameter(
            "n", "int", 1,
            display_name=tr_("Repeats"),
            min_value=1
        )
    
    def build_keras_layer(self):
        from tensorflow import keras
        
        return keras.layers.RepeatVector(
            n=self.get_parameter("n"),
            name=self.layer_name or None
        )


@register_layer  
class UpSampling1DLayer(LayerNode):
    """1D上采样层"""
    layer_type = "upsampling1d"
    display_name = "UpSampling1D"
    category = LayerCategory.RESHAPE
    description = tr_("1D upsampling layer")
    icon = "⬆️"
    keras_class = "UpSampling1D"
    
    def _setup_parameters(self):
        self.add_parameter(
            "size", "int", 2,
            display_name=tr_("Upsampling factor"),
            min_value=1
        )
    
    def build_keras_layer(self):
        from tensorflow import keras
        
        return keras.layers.UpSampling1D(
            size=self.get_parameter("size"),
            name=self.layer_name or None
        )


@register_layer
class UpSampling2DLayer(LayerNode):
    """2D上采样层"""
    layer_type = "upsampling2d"
    display_name = "UpSampling2D"
    category = LayerCategory.RESHAPE
    description = tr_("2D upsampling layer")
    icon = "⬆️"
    keras_class = "UpSampling2D"
    
    def _setup_parameters(self):
        self.add_parameter(
            "size", "str", "(2, 2)",
            display_name=tr_("Upsampling factor")
        )
    
    def build_keras_layer(self):
        from tensorflow import keras
        
        size = eval(self.get_parameter("size"))
        return keras.layers.UpSampling2D(
            size=size,
            name=self.layer_name or None
        )

