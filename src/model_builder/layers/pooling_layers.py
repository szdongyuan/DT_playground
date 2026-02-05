# -*- coding: utf-8 -*-
"""
Pooling Layer Nodes
"""

from ..layer_base import LayerCategory, LayerNode, register_layer
from src.ui.i18n import tr_
@register_layer
class MaxPooling1DLayer(LayerNode):
    """1D最大池化层"""
    layer_type = "max_pooling1d"
    display_name = "MaxPooling1D"
    category = LayerCategory.POOLING
    description = tr_("1D max pooling layer")
    icon = "⬇️"
    keras_class = "MaxPooling1D"
    
    def _setup_parameters(self):
        self.add_parameter(
            "pool_size", "int", 2,
            display_name=tr_("Pool size"),
            min_value=1
        )
        self.add_parameter(
            "strides", "int", 2,
            display_name=tr_("Stride"),
            min_value=1,
            required=False
        )
        self.add_parameter(
            "padding", "choice", "valid",
            display_name=tr_("Padding"),
            choices=["valid", "same"]
        )
    
    def build_keras_layer(self):
        from tensorflow import keras
        
        strides = self.get_parameter("strides")
        return keras.layers.MaxPooling1D(
            pool_size=self.get_parameter("pool_size"),
            strides=strides if strides else None,
            padding=self.get_parameter("padding"),
            name=self.layer_name or None
        )


@register_layer
class MaxPooling2DLayer(LayerNode):
    """2D最大池化层"""
    layer_type = "max_pooling2d"
    display_name = "MaxPooling2D"
    category = LayerCategory.POOLING
    description = tr_("2D max pooling layer")
    icon = "⬇️"
    keras_class = "MaxPooling2D"
    
    def _setup_parameters(self):
        self.add_parameter(
            "pool_size", "str", "(2, 2)",
            display_name=tr_("Pool size")
        )
        self.add_parameter(
            "strides", "str", "(2, 2)",
            display_name=tr_("Stride")
        )
        self.add_parameter(
            "padding", "choice", "valid",
            display_name=tr_("Padding"),
            choices=["valid", "same"]
        )
    
    def build_keras_layer(self):
        from tensorflow import keras
        
        pool_size = eval(self.get_parameter("pool_size"))
        strides = eval(self.get_parameter("strides"))
        
        return keras.layers.MaxPooling2D(
            pool_size=pool_size,
            strides=strides,
            padding=self.get_parameter("padding"),
            name=self.layer_name or None
        )


@register_layer
class AveragePooling1DLayer(LayerNode):
    """1D平均池化层"""
    layer_type = "avg_pooling1d"
    display_name = "AveragePooling1D"
    category = LayerCategory.POOLING
    description = tr_("1D average pooling layer")
    icon = "📉"
    keras_class = "AveragePooling1D"
    
    def _setup_parameters(self):
        self.add_parameter(
            "pool_size", "int", 2,
            display_name=tr_("Pool size"),
            min_value=1
        )
        self.add_parameter(
            "strides", "int", 2,
            display_name=tr_("Stride"),
            min_value=1
        )
        self.add_parameter(
            "padding", "choice", "valid",
            display_name=tr_("Padding"),
            choices=["valid", "same"]
        )
    
    def build_keras_layer(self):
        from tensorflow import keras
        
        return keras.layers.AveragePooling1D(
            pool_size=self.get_parameter("pool_size"),
            strides=self.get_parameter("strides"),
            padding=self.get_parameter("padding"),
            name=self.layer_name or None
        )


@register_layer
class GlobalMaxPooling1DLayer(LayerNode):
    """全局1D最大池化层"""
    layer_type = "global_max_pooling1d"
    display_name = "GlobalMaxPooling1D"
    category = LayerCategory.POOLING
    description = tr_("Global 1D max pooling layer")
    icon = "🔽"
    keras_class = "GlobalMaxPooling1D"
    
    def _setup_parameters(self):
        pass  # 无参数
    
    def build_keras_layer(self):
        from tensorflow import keras
        return keras.layers.GlobalMaxPooling1D(name=self.layer_name or None)


@register_layer
class GlobalAveragePooling1DLayer(LayerNode):
    """全局1D平均池化层"""
    layer_type = "global_avg_pooling1d"
    display_name = "GlobalAveragePooling1D"
    category = LayerCategory.POOLING
    description = tr_("Global 1D average pooling layer")
    icon = "📊"
    keras_class = "GlobalAveragePooling1D"
    
    def _setup_parameters(self):
        pass  # 无参数
    
    def build_keras_layer(self):
        from tensorflow import keras
        return keras.layers.GlobalAveragePooling1D(name=self.layer_name or None)


@register_layer
class GlobalMaxPooling2DLayer(LayerNode):
    """全局2D最大池化层"""
    layer_type = "global_max_pooling2d"
    display_name = "GlobalMaxPooling2D"
    category = LayerCategory.POOLING
    description = tr_("Global 2D max pooling layer")
    icon = "🔽"
    keras_class = "GlobalMaxPooling2D"
    
    def _setup_parameters(self):
        pass
    
    def build_keras_layer(self):
        from tensorflow import keras
        return keras.layers.GlobalMaxPooling2D(name=self.layer_name or None)


@register_layer
class GlobalAveragePooling2DLayer(LayerNode):
    """全局2D平均池化层"""
    layer_type = "global_avg_pooling2d"
    display_name = "GlobalAveragePooling2D"
    category = LayerCategory.POOLING
    description = tr_("Global 2D average pooling layer")
    icon = "📊"
    keras_class = "GlobalAveragePooling2D"
    
    def _setup_parameters(self):
        pass
    
    def build_keras_layer(self):
        from tensorflow import keras
        return keras.layers.GlobalAveragePooling2D(name=self.layer_name or None)

