# -*- coding: utf-8 -*-
"""
Merge Layer Nodes
"""

from ..layer_base import LayerCategory, LayerNode, register_layer
from src.ui.i18n import tr_
@register_layer
class ConcatenateLayer(LayerNode):
    """连接层"""
    layer_type = "concatenate"
    display_name = tr_("Concatenate")
    category = LayerCategory.MERGE
    description = tr_("Concatenate a list of inputs along a given axis")
    icon = "🔗"
    keras_class = "Concatenate"
    
    def _setup_parameters(self):
        self.add_parameter(
            "axis", "int", -1,
            display_name=tr_("Axis"),
            description=tr_("Axis to concatenate")
        )
    
    def build_keras_layer(self):
        from tensorflow import keras
        
        return keras.layers.Concatenate(
            axis=self.get_parameter("axis"),
            name=self.layer_name or None
        )


@register_layer
class AddLayer(LayerNode):
    """加法层"""
    layer_type = "add"
    display_name = tr_("Add")
    category = LayerCategory.MERGE
    description = tr_("Element-wise addition of a list of inputs")
    icon = "➕"
    keras_class = "Add"
    
    def _setup_parameters(self):
        pass  # 无参数
    
    def build_keras_layer(self):
        from tensorflow import keras
        return keras.layers.Add(name=self.layer_name or None)


@register_layer
class MultiplyLayer(LayerNode):
    """乘法层"""
    layer_type = "multiply"
    display_name = tr_("Multiply")
    category = LayerCategory.MERGE
    description = tr_("Element-wise multiplication of a list of inputs")
    icon = "✖️"
    keras_class = "Multiply"
    
    def _setup_parameters(self):
        pass
    
    def build_keras_layer(self):
        from tensorflow import keras
        return keras.layers.Multiply(name=self.layer_name or None)


@register_layer
class AverageLayer(LayerNode):
    """平均层"""
    layer_type = "average"
    display_name = tr_("Average")
    category = LayerCategory.MERGE
    description = tr_("Element-wise average of a list of inputs")
    icon = "📊"
    keras_class = "Average"
    
    def _setup_parameters(self):
        pass
    
    def build_keras_layer(self):
        from tensorflow import keras
        return keras.layers.Average(name=self.layer_name or None)


@register_layer
class MaximumLayer(LayerNode):
    """最大值层"""
    layer_type = "maximum"
    display_name = tr_("Maximum")
    category = LayerCategory.MERGE
    description = tr_("Element-wise maximum of a list of inputs")
    icon = "📈"
    keras_class = "Maximum"
    
    def _setup_parameters(self):
        pass
    
    def build_keras_layer(self):
        from tensorflow import keras
        return keras.layers.Maximum(name=self.layer_name or None)


@register_layer
class MinimumLayer(LayerNode):
    """最小值层"""
    layer_type = "minimum"
    display_name = tr_("Minimum")
    category = LayerCategory.MERGE
    description = tr_("Element-wise minimum of a list of inputs")
    icon = "📉"
    keras_class = "Minimum"
    
    def _setup_parameters(self):
        pass
    
    def build_keras_layer(self):
        from tensorflow import keras
        return keras.layers.Minimum(name=self.layer_name or None)


@register_layer
class DotLayer(LayerNode):
    """点积层"""
    layer_type = "dot"
    display_name = tr_("Dot")
    category = LayerCategory.MERGE
    description = tr_("Compute the dot product of two inputs")
    icon = "⚫"
    keras_class = "Dot"
    
    def _setup_parameters(self):
        self.add_parameter(
            "axes", "int", -1,
            display_name=tr_("Axes"),
            description=tr_("Axes for dot product")
        )
        self.add_parameter(
            "normalize", "bool", False,
            display_name=tr_("Normalize"),
            description=tr_("Whether to apply L2 normalization before dot product")
        )
    
    def build_keras_layer(self):
        from tensorflow import keras
        
        return keras.layers.Dot(
            axes=self.get_parameter("axes"),
            normalize=self.get_parameter("normalize"),
            name=self.layer_name or None
        )

