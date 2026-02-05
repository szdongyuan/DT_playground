# -*- coding: utf-8 -*-
"""
Normalization Layer Nodes
"""

from ..layer_base import LayerCategory, LayerNode, register_layer
from src.ui.i18n import tr_
@register_layer
class BatchNormalizationLayer(LayerNode):
    """批归一化层"""
    layer_type = "batch_norm"
    display_name = "BatchNormalization"
    category = LayerCategory.NORMALIZATION
    description = tr_("Batch normalization layer")
    icon = "📐"
    keras_class = "BatchNormalization"
    
    def _setup_parameters(self):
        self.add_parameter(
            "momentum", "float", 0.99,
            display_name=tr_("Momentum"),
            description=tr_("Momentum for the moving average"),
            min_value=0.0,
            max_value=1.0
        )
        self.add_parameter(
            "epsilon", "float", 0.001,
            display_name="Epsilon",
            description=tr_("Small float added to variance to avoid division by zero"),
            min_value=1e-7
        )
        self.add_parameter(
            "center", "bool", True,
            display_name=tr_("Center"),
            description=tr_("Whether to add beta offset")
        )
        self.add_parameter(
            "scale", "bool", True,
            display_name=tr_("Scale"),
            description=tr_("Whether to multiply by gamma")
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
    description = tr_("Layer normalization")
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
            display_name=tr_("Center")
        )
        self.add_parameter(
            "scale", "bool", True,
            display_name=tr_("Scale")
        )
    
    def build_keras_layer(self):
        from tensorflow import keras
        
        return keras.layers.LayerNormalization(
            epsilon=self.get_parameter("epsilon"),
            center=self.get_parameter("center"),
            scale=self.get_parameter("scale"),
            name=self.layer_name or None
        )

