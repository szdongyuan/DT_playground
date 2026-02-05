# -*- coding: utf-8 -*-
"""
Regularization Layer Nodes (Dropout, etc.)
"""

from ..layer_base import LayerCategory, LayerNode, register_layer
from src.ui.i18n import tr_
@register_layer
class DropoutLayer(LayerNode):
    """Dropout层"""
    layer_type = "dropout"
    display_name = "Dropout"
    category = LayerCategory.REGULARIZATION
    description = tr_("Randomly drops input units to reduce overfitting")
    icon = "💧"
    keras_class = "Dropout"
    
    def _setup_parameters(self):
        self.add_parameter(
            "rate", "float", 0.5,
            display_name=tr_("Dropout rate"),
            description=tr_("Fraction of the input units to drop"),
            min_value=0.0,
            max_value=1.0
        )
    
    def build_keras_layer(self):
        from tensorflow import keras
        
        return keras.layers.Dropout(
            rate=self.get_parameter("rate"),
            name=self.layer_name or None
        )


@register_layer
class SpatialDropout1DLayer(LayerNode):
    """空间Dropout1D层"""
    layer_type = "spatial_dropout1d"
    display_name = "SpatialDropout1D"
    category = LayerCategory.REGULARIZATION
    description = tr_("Spatial 1D dropout (drops entire feature maps)")
    icon = "💦"
    keras_class = "SpatialDropout1D"
    
    def _setup_parameters(self):
        self.add_parameter(
            "rate", "float", 0.5,
            display_name=tr_("Dropout rate"),
            min_value=0.0,
            max_value=1.0
        )
    
    def build_keras_layer(self):
        from tensorflow import keras
        
        return keras.layers.SpatialDropout1D(
            rate=self.get_parameter("rate"),
            name=self.layer_name or None
        )


@register_layer
class SpatialDropout2DLayer(LayerNode):
    """空间Dropout2D层"""
    layer_type = "spatial_dropout2d"
    display_name = "SpatialDropout2D"
    category = LayerCategory.REGULARIZATION
    description = tr_("Spatial 2D dropout")
    icon = "💦"
    keras_class = "SpatialDropout2D"
    
    def _setup_parameters(self):
        self.add_parameter(
            "rate", "float", 0.5,
            display_name=tr_("Dropout rate"),
            min_value=0.0,
            max_value=1.0
        )
    
    def build_keras_layer(self):
        from tensorflow import keras
        
        return keras.layers.SpatialDropout2D(
            rate=self.get_parameter("rate"),
            name=self.layer_name or None
        )


@register_layer
class GaussianNoiseLayer(LayerNode):
    """高斯噪声层"""
    layer_type = "gaussian_noise"
    display_name = "GaussianNoise"
    category = LayerCategory.REGULARIZATION
    description = tr_("Applies additive Gaussian noise")
    icon = "🌫️"
    keras_class = "GaussianNoise"
    
    def _setup_parameters(self):
        self.add_parameter(
            "stddev", "float", 0.1,
            display_name=tr_("Standard deviation"),
            description=tr_("Standard deviation of the noise distribution"),
            min_value=0.0
        )
    
    def build_keras_layer(self):
        from tensorflow import keras
        
        return keras.layers.GaussianNoise(
            stddev=self.get_parameter("stddev"),
            name=self.layer_name or None
        )


@register_layer
class GaussianDropoutLayer(LayerNode):
    """高斯Dropout层"""
    layer_type = "gaussian_dropout"
    display_name = "GaussianDropout"
    category = LayerCategory.REGULARIZATION
    description = tr_("Applies multiplicative Gaussian noise")
    icon = "🌫️"
    keras_class = "GaussianDropout"
    
    def _setup_parameters(self):
        self.add_parameter(
            "rate", "float", 0.5,
            display_name=tr_("Dropout rate"),
            min_value=0.0,
            max_value=1.0
        )
    
    def build_keras_layer(self):
        from tensorflow import keras
        
        return keras.layers.GaussianDropout(
            rate=self.get_parameter("rate"),
            name=self.layer_name or None
        )

