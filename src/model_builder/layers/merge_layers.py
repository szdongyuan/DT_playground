# -*- coding: utf-8 -*-
"""
Merge Layer Nodes
"""

from ..layer_base import LayerCategory, LayerNode, register_layer


@register_layer
class ConcatenateLayer(LayerNode):
    """连接层"""
    layer_type = "concatenate"
    display_name = "Concatenate 连接"
    category = LayerCategory.MERGE
    description = "沿指定轴连接输入列表"
    icon = "🔗"
    keras_class = "Concatenate"
    
    def _setup_parameters(self):
        self.add_parameter(
            "axis", "int", -1,
            display_name="连接轴",
            description="连接的轴"
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
    display_name = "Add 加法"
    category = LayerCategory.MERGE
    description = "对输入列表进行逐元素加法"
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
    display_name = "Multiply 乘法"
    category = LayerCategory.MERGE
    description = "对输入列表进行逐元素乘法"
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
    display_name = "Average 平均"
    category = LayerCategory.MERGE
    description = "对输入列表进行逐元素平均"
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
    display_name = "Maximum 最大值"
    category = LayerCategory.MERGE
    description = "对输入列表进行逐元素取最大值"
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
    display_name = "Minimum 最小值"
    category = LayerCategory.MERGE
    description = "对输入列表进行逐元素取最小值"
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
    display_name = "Dot 点积"
    category = LayerCategory.MERGE
    description = "计算两个张量的点积"
    icon = "⚫"
    keras_class = "Dot"
    
    def _setup_parameters(self):
        self.add_parameter(
            "axes", "int", -1,
            display_name="点积轴",
            description="进行点积的轴"
        )
        self.add_parameter(
            "normalize", "bool", False,
            display_name="归一化",
            description="是否在点积前L2归一化"
        )
    
    def build_keras_layer(self):
        from tensorflow import keras
        
        return keras.layers.Dot(
            axes=self.get_parameter("axes"),
            normalize=self.get_parameter("normalize"),
            name=self.layer_name or None
        )

