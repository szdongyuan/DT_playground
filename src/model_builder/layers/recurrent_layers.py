# -*- coding: utf-8 -*-
"""
循环层节点 (LSTM, GRU等)
"""

from ..layer_base import LayerCategory, LayerNode, register_layer


@register_layer
class LSTMLayer(LayerNode):
    """LSTM层"""
    layer_type = "lstm"
    display_name = "LSTM"
    category = LayerCategory.RECURRENT
    description = "长短期记忆网络层"
    icon = "🔄"
    keras_class = "LSTM"
    
    def _setup_parameters(self):
        self.add_parameter(
            "units", "int", 64,
            display_name="单元数量",
            description="输出空间的维度",
            min_value=1
        )
        self.add_parameter(
            "return_sequences", "bool", False,
            display_name="返回序列",
            description="是否返回输出序列中的最后一个输出，还是完整序列"
        )
        self.add_parameter(
            "return_state", "bool", False,
            display_name="返回状态",
            description="是否返回最后一个状态"
        )
        self.add_parameter(
            "activation", "choice", "tanh",
            display_name="激活函数",
            choices=["tanh", "sigmoid", "relu", "linear"],
            description="激活函数"
        )
        self.add_parameter(
            "recurrent_activation", "choice", "sigmoid",
            display_name="循环激活函数",
            choices=["sigmoid", "hard_sigmoid", "tanh"],
            description="用于循环步骤的激活函数"
        )
        self.add_parameter(
            "dropout", "float", 0.0,
            display_name="Dropout率",
            description="输入的线性变换的dropout比率",
            min_value=0.0,
            max_value=1.0
        )
        self.add_parameter(
            "recurrent_dropout", "float", 0.0,
            display_name="循环Dropout率",
            description="循环状态的线性变换的dropout比率",
            min_value=0.0,
            max_value=1.0
        )
        self.add_parameter(
            "bidirectional", "bool", False,
            display_name="双向",
            description="是否使用双向LSTM"
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
    description = "门控循环单元层"
    icon = "🔁"
    keras_class = "GRU"
    
    def _setup_parameters(self):
        self.add_parameter(
            "units", "int", 64,
            display_name="单元数量",
            min_value=1
        )
        self.add_parameter(
            "return_sequences", "bool", False,
            display_name="返回序列"
        )
        self.add_parameter(
            "activation", "choice", "tanh",
            display_name="激活函数",
            choices=["tanh", "sigmoid", "relu", "linear"]
        )
        self.add_parameter(
            "dropout", "float", 0.0,
            display_name="Dropout率",
            min_value=0.0,
            max_value=1.0
        )
        self.add_parameter(
            "recurrent_dropout", "float", 0.0,
            display_name="循环Dropout率",
            min_value=0.0,
            max_value=1.0
        )
        self.add_parameter(
            "bidirectional", "bool", False,
            display_name="双向"
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
    description = "简单循环神经网络层"
    icon = "➰"
    keras_class = "SimpleRNN"
    
    def _setup_parameters(self):
        self.add_parameter(
            "units", "int", 64,
            display_name="单元数量",
            min_value=1
        )
        self.add_parameter(
            "return_sequences", "bool", False,
            display_name="返回序列"
        )
        self.add_parameter(
            "activation", "choice", "tanh",
            display_name="激活函数",
            choices=["tanh", "sigmoid", "relu", "linear"]
        )
        self.add_parameter(
            "dropout", "float", 0.0,
            display_name="Dropout率",
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

