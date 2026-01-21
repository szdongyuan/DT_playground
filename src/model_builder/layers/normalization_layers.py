# -*- coding: utf-8 -*-
"""
Normalization Layer Nodes
"""

from ..layer_base import LayerCategory, LayerNode, register_layer


@register_layer
class BatchNormalizationLayer(LayerNode):
    """批归一化层"""
    layer_type = "batch_norm"
    display_name = "BatchNormalization"
    category = LayerCategory.NORMALIZATION
    description = "批量归一化层"
    icon = "📐"
    keras_class = "BatchNormalization"
    
    def _setup_parameters(self):
        self.add_parameter(
            "momentum", "float", 0.99,
            display_name="动量",
            description="移动平均的动量",
            min_value=0.0,
            max_value=1.0
        )
        self.add_parameter(
            "epsilon", "float", 0.001,
            display_name="Epsilon",
            description="添加到方差的小浮点数，避免除零",
            min_value=1e-7
        )
        self.add_parameter(
            "center", "bool", True,
            display_name="中心化",
            description="是否添加beta偏置"
        )
        self.add_parameter(
            "scale", "bool", True,
            display_name="缩放",
            description="是否乘以gamma"
        )
    
    def build_keras_layer(self):
        from tensorflow import keras
        
        return keras.layers.BatchNormalization(
            momentum=self.get_parameter("momentum"),
            epsilon=self.get_parameter("epsilon"),
            center=self.get_parameter("center"),
            scale=self.get_parameter("scale"),
            name=self.layer_name or None
        )


@register_layer
class LayerNormalizationLayer(LayerNode):
    """层归一化"""
    layer_type = "layer_norm"
    display_name = "LayerNormalization"
    category = LayerCategory.NORMALIZATION
    description = "层归一化"
    icon = "📏"
    keras_class = "LayerNormalization"
    
    def _setup_parameters(self):
        self.add_parameter(
            "epsilon", "float", 0.001,
            display_name="Epsilon",
            min_value=1e-7
        )
        self.add_parameter(
            "center", "bool", True,
            display_name="中心化"
        )
        self.add_parameter(
            "scale", "bool", True,
            display_name="缩放"
        )
    
    def build_keras_layer(self):
        from tensorflow import keras
        
        return keras.layers.LayerNormalization(
            epsilon=self.get_parameter("epsilon"),
            center=self.get_parameter("center"),
            scale=self.get_parameter("scale"),
            name=self.layer_name or None
        )

