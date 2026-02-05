# -*- coding: utf-8 -*-
"""
Recurrent Layer Nodes (LSTM, GRU, etc.)
"""

from ..layer_base import LayerCategory, LayerNode, register_layer
from src.ui.i18n import tr_
@register_layer
class LSTMLayer(LayerNode):
    """LSTM层"""
    layer_type = "lstm"
    display_name = "LSTM"
    category = LayerCategory.RECURRENT
    description = tr_("Long Short-Term Memory (LSTM) layer")
    icon = "🔄"
    keras_class = "LSTM"
    
    def _setup_parameters(self):
        self.add_parameter(
            "units", "int", 64,
            display_name=tr_("Units"),
            description=tr_("Dimensionality of the output space"),
            min_value=1
        )
        self.add_parameter(
            "return_sequences", "bool", False,
            display_name=tr_("Return sequences"),
            description=tr_("Whether to return the full sequence or only the last output")
        )
        self.add_parameter(
            "return_state", "bool", False,
            display_name=tr_("Return state"),
            description=tr_("Whether to return the last state")
        )
        self.add_parameter(
            "activation", "choice", "tanh",
            display_name=tr_("Activation"),
            choices=["tanh", "sigmoid", "relu", "linear"],
            description=tr_("Activation function")
        )
        self.add_parameter(
            "recurrent_activation", "choice", "sigmoid",
            display_name=tr_("Recurrent activation"),
            choices=["sigmoid", "hard_sigmoid", "tanh"],
            description=tr_("Activation function for the recurrent step")
        )
        self.add_parameter(
            "dropout", "float", 0.0,
            display_name=tr_("Dropout rate"),
            description=tr_("Dropout rate for the input linear transformation"),
            min_value=0.0,
            max_value=1.0
        )
        self.add_parameter(
            "recurrent_dropout", "float", 0.0,
            display_name=tr_("Recurrent dropout rate"),
            description=tr_("Dropout rate for the recurrent state linear transformation"),
            min_value=0.0,
            max_value=1.0
        )
        self.add_parameter(
            "bidirectional", "bool", False,
            display_name=tr_("Bidirectional"),
            description=tr_("Whether to use a bidirectional wrapper")
        )
    
    def build_keras_layer(self):
        from tensorflow import keras
        
        lstm = keras.layers.LSTM(
            units=self.get_parameter("units"),
            return_sequences=self.get_parameter("return_sequences"),
            return_state=self.get_parameter("return_state"),
            activation=self.get_parameter("activation"),
            recurrent_activation=self.get_parameter("recurrent_activation"),
            dropout=self.get_parameter("dropout"),
            recurrent_dropout=self.get_parameter("recurrent_dropout"),
            name=self.layer_name or None
        )
        
        if self.get_parameter("bidirectional"):
            return keras.layers.Bidirectional(lstm)
        return lstm


@register_layer
class GRULayer(LayerNode):
    """GRU层"""
    layer_type = "gru"
    display_name = "GRU"
    category = LayerCategory.RECURRENT
    description = tr_("Gated Recurrent Unit (GRU) layer")
    icon = "🔁"
    keras_class = "GRU"
    
    def _setup_parameters(self):
        self.add_parameter(
            "units", "int", 64,
            display_name=tr_("Units"),
            min_value=1
        )
        self.add_parameter(
            "return_sequences", "bool", False,
            display_name=tr_("Return sequences")
        )
        self.add_parameter(
            "activation", "choice", "tanh",
            display_name=tr_("Activation"),
            choices=["tanh", "sigmoid", "relu", "linear"]
        )
        self.add_parameter(
            "dropout", "float", 0.0,
            display_name=tr_("Dropout rate"),
            min_value=0.0,
            max_value=1.0
        )
        self.add_parameter(
            "recurrent_dropout", "float", 0.0,
            display_name=tr_("Recurrent dropout rate"),
            min_value=0.0,
            max_value=1.0
        )
        self.add_parameter(
            "bidirectional", "bool", False,
            display_name=tr_("Bidirectional")
        )
    
    def build_keras_layer(self):
        from tensorflow import keras
        
        gru = keras.layers.GRU(
            units=self.get_parameter("units"),
            return_sequences=self.get_parameter("return_sequences"),
            activation=self.get_parameter("activation"),
            dropout=self.get_parameter("dropout"),
            recurrent_dropout=self.get_parameter("recurrent_dropout"),
            name=self.layer_name or None
        )
        
        if self.get_parameter("bidirectional"):
            return keras.layers.Bidirectional(gru)
        return gru


@register_layer
class SimpleRNNLayer(LayerNode):
    """简单RNN层"""
    layer_type = "simple_rnn"
    display_name = "SimpleRNN"
    category = LayerCategory.RECURRENT
    description = tr_("Simple RNN layer")
    icon = "➰"
    keras_class = "SimpleRNN"
    
    def _setup_parameters(self):
        self.add_parameter(
            "units", "int", 64,
            display_name=tr_("Units"),
            min_value=1
        )
        self.add_parameter(
            "return_sequences", "bool", False,
            display_name=tr_("Return sequences")
        )
        self.add_parameter(
            "activation", "choice", "tanh",
            display_name=tr_("Activation"),
            choices=["tanh", "sigmoid", "relu", "linear"]
        )
        self.add_parameter(
            "dropout", "float", 0.0,
            display_name=tr_("Dropout rate"),
            min_value=0.0,
            max_value=1.0
        )
    
    def build_keras_layer(self):
        from tensorflow import keras
        
        return keras.layers.SimpleRNN(
            units=self.get_parameter("units"),
            return_sequences=self.get_parameter("return_sequences"),
            activation=self.get_parameter("activation"),
            dropout=self.get_parameter("dropout"),
            name=self.layer_name or None
        )

