# -*- coding: utf-8 -*-
"""
注意力层节点 (MultiHeadAttention, TransformerEncoder等)

实现 Transformer 架构相关的层节点。
"""

from ..layer_base import LayerCategory, LayerNode, register_layer


@register_layer
class MultiHeadAttentionLayer(LayerNode):
    """多头注意力层"""
    layer_type = "multi_head_attention"
    display_name = "MultiHeadAttention"
    category = LayerCategory.ATTENTION
    description = "多头注意力机制层，用于序列建模"
    icon = "🎯"
    keras_class = "MultiHeadAttention"
    
    def _setup_parameters(self):
        self.add_parameter(
            "num_heads", "int", 8,
            display_name="注意力头数",
            description="多头注意力的头数量",
            min_value=1,
            max_value=64
        )
        self.add_parameter(
            "key_dim", "int", 64,
            display_name="键维度",
            description="每个注意力头的查询和键的维度",
            min_value=1
        )
        self.add_parameter(
            "value_dim", "int", 0,
            display_name="值维度",
            description="每个注意力头的值的维度，0表示等于key_dim",
            min_value=0,
            required=False
        )
        self.add_parameter(
            "dropout", "float", 0.0,
            display_name="Dropout率",
            description="注意力权重的dropout比率",
            min_value=0.0,
            max_value=1.0
        )
        self.add_parameter(
            "use_bias", "bool", True,
            display_name="使用偏置",
            description="是否使用偏置向量"
        )
    
    def build_keras_layer(self):
        from tensorflow import keras
        
        value_dim = self.get_parameter("value_dim")
        if value_dim == 0:
            value_dim = None  # 使用key_dim作为默认值
        
        return keras.layers.MultiHeadAttention(
            num_heads=self.get_parameter("num_heads"),
            key_dim=self.get_parameter("key_dim"),
            value_dim=value_dim,
            dropout=self.get_parameter("dropout"),
            use_bias=self.get_parameter("use_bias"),
            name=self.layer_name or None
        )


@register_layer
class TransformerEncoderLayer(LayerNode):
    """
    Transformer编码器块
    
    包含多头自注意力 + 前馈网络 + 残差连接 + 层归一化
    """
    layer_type = "transformer_encoder"
    display_name = "TransformerEncoder"
    category = LayerCategory.ATTENTION
    description = "Transformer编码器块（自注意力+FFN+残差+归一化）"
    icon = "🔀"
    keras_class = "TransformerEncoder"
    
    def _setup_parameters(self):
        self.add_parameter(
            "num_heads", "int", 8,
            display_name="注意力头数",
            description="多头注意力的头数量",
            min_value=1,
            max_value=64
        )
        self.add_parameter(
            "key_dim", "int", 64,
            display_name="键维度",
            description="每个注意力头的查询和键的维度",
            min_value=1
        )
        self.add_parameter(
            "ff_dim", "int", 256,
            display_name="前馈网络维度",
            description="前馈网络的中间层维度",
            min_value=1
        )
        self.add_parameter(
            "dropout_rate", "float", 0.1,
            display_name="Dropout率",
            description="Dropout比率",
            min_value=0.0,
            max_value=1.0
        )
        self.add_parameter(
            "activation", "choice", "gelu",
            display_name="激活函数",
            description="前馈网络的激活函数",
            choices=["gelu", "relu", "swish", "silu", "tanh"]
        )
        self.add_parameter(
            "epsilon", "float", 1e-6,
            display_name="归一化Epsilon",
            description="层归一化的epsilon参数",
            min_value=1e-10,
            max_value=1e-3
        )
        self.add_parameter(
            "pre_norm", "bool", False,
            display_name="Pre-Norm",
            description="使用Pre-Norm（先归一化再注意力），否则使用Post-Norm"
        )
    
    def build_keras_layer(self):
        """
        构建 Transformer Encoder 块
        
        由于 Keras 没有内置的 TransformerEncoder，这里返回一个自定义的 Lambda 层
        或者使用 Keras 的 TransformerEncoder（如果可用）
        """
        from tensorflow import keras
        import tensorflow as tf
        
        num_heads = self.get_parameter("num_heads")
        key_dim = self.get_parameter("key_dim")
        ff_dim = self.get_parameter("ff_dim")
        dropout_rate = self.get_parameter("dropout_rate")
        activation = self.get_parameter("activation")
        epsilon = self.get_parameter("epsilon")
        pre_norm = self.get_parameter("pre_norm")
        layer_name = self.layer_name or "transformer_encoder"
        
        # 创建自定义 TransformerEncoder 层
        class TransformerEncoderBlock(keras.layers.Layer):
            def __init__(self, num_heads, key_dim, ff_dim, dropout_rate, 
                         activation, epsilon, pre_norm, **kwargs):
                super().__init__(**kwargs)
                self.num_heads = num_heads
                self.key_dim = key_dim
                self.ff_dim = ff_dim
                self.dropout_rate = dropout_rate
                self.activation = activation
                self.epsilon = epsilon
                self.pre_norm = pre_norm
                
                # 多头注意力
                self.mha = keras.layers.MultiHeadAttention(
                    num_heads=num_heads,
                    key_dim=key_dim,
                    dropout=dropout_rate
                )
                
                # 前馈网络
                self.ffn = keras.Sequential([
                    keras.layers.Dense(ff_dim, activation=activation),
                    keras.layers.Dropout(dropout_rate),
                    keras.layers.Dense(key_dim * num_heads),  # 恢复原始维度
                    keras.layers.Dropout(dropout_rate)
                ])
                
                # 层归一化
                self.layernorm1 = keras.layers.LayerNormalization(epsilon=epsilon)
                self.layernorm2 = keras.layers.LayerNormalization(epsilon=epsilon)
                
                # Dropout
                self.dropout1 = keras.layers.Dropout(dropout_rate)
            
            def call(self, inputs, training=None):
                if self.pre_norm:
                    # Pre-Norm: LayerNorm -> Attention -> Residual
                    x = self.layernorm1(inputs)
                    attn_output = self.mha(x, x, training=training)
                    attn_output = self.dropout1(attn_output, training=training)
                    out1 = inputs + attn_output
                    
                    x = self.layernorm2(out1)
                    ffn_output = self.ffn(x, training=training)
                    return out1 + ffn_output
                else:
                    # Post-Norm: Attention -> Residual -> LayerNorm
                    attn_output = self.mha(inputs, inputs, training=training)
                    attn_output = self.dropout1(attn_output, training=training)
                    out1 = self.layernorm1(inputs + attn_output)
                    
                    ffn_output = self.ffn(out1, training=training)
                    return self.layernorm2(out1 + ffn_output)
            
            def get_config(self):
                config = super().get_config()
                config.update({
                    "num_heads": self.num_heads,
                    "key_dim": self.key_dim,
                    "ff_dim": self.ff_dim,
                    "dropout_rate": self.dropout_rate,
                    "activation": self.activation,
                    "epsilon": self.epsilon,
                    "pre_norm": self.pre_norm
                })
                return config
        
        return TransformerEncoderBlock(
            num_heads=num_heads,
            key_dim=key_dim,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate,
            activation=activation,
            epsilon=epsilon,
            pre_norm=pre_norm,
            name=layer_name
        )


@register_layer
class TransformerDecoderLayer(LayerNode):
    """
    Transformer解码器块
    
    包含自注意力 + 交叉注意力 + 前馈网络 + 残差连接 + 层归一化
    """
    layer_type = "transformer_decoder"
    display_name = "TransformerDecoder"
    category = LayerCategory.ATTENTION
    description = "Transformer解码器块（自注意力+交叉注意力+FFN）"
    icon = "🔁"
    keras_class = "TransformerDecoder"
    
    def _setup_parameters(self):
        self.add_parameter(
            "num_heads", "int", 8,
            display_name="注意力头数",
            description="多头注意力的头数量",
            min_value=1,
            max_value=64
        )
        self.add_parameter(
            "key_dim", "int", 64,
            display_name="键维度",
            description="每个注意力头的查询和键的维度",
            min_value=1
        )
        self.add_parameter(
            "ff_dim", "int", 256,
            display_name="前馈网络维度",
            description="前馈网络的中间层维度",
            min_value=1
        )
        self.add_parameter(
            "dropout_rate", "float", 0.1,
            display_name="Dropout率",
            description="Dropout比率",
            min_value=0.0,
            max_value=1.0
        )
        self.add_parameter(
            "activation", "choice", "gelu",
            display_name="激活函数",
            description="前馈网络的激活函数",
            choices=["gelu", "relu", "swish", "silu", "tanh"]
        )
    
    def build_keras_layer(self):
        from tensorflow import keras
        
        num_heads = self.get_parameter("num_heads")
        key_dim = self.get_parameter("key_dim")
        ff_dim = self.get_parameter("ff_dim")
        dropout_rate = self.get_parameter("dropout_rate")
        activation = self.get_parameter("activation")
        layer_name = self.layer_name or "transformer_decoder"
        
        class TransformerDecoderBlock(keras.layers.Layer):
            def __init__(self, num_heads, key_dim, ff_dim, dropout_rate, 
                         activation, **kwargs):
                super().__init__(**kwargs)
                self.num_heads = num_heads
                self.key_dim = key_dim
                self.ff_dim = ff_dim
                self.dropout_rate = dropout_rate
                self.activation = activation
                
                # 自注意力（带因果掩码）
                self.self_attention = keras.layers.MultiHeadAttention(
                    num_heads=num_heads,
                    key_dim=key_dim,
                    dropout=dropout_rate
                )
                
                # 交叉注意力
                self.cross_attention = keras.layers.MultiHeadAttention(
                    num_heads=num_heads,
                    key_dim=key_dim,
                    dropout=dropout_rate
                )
                
                # 前馈网络
                self.ffn = keras.Sequential([
                    keras.layers.Dense(ff_dim, activation=activation),
                    keras.layers.Dropout(dropout_rate),
                    keras.layers.Dense(key_dim * num_heads),
                    keras.layers.Dropout(dropout_rate)
                ])
                
                # 层归一化
                self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
                self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
                self.layernorm3 = keras.layers.LayerNormalization(epsilon=1e-6)
                
                self.dropout1 = keras.layers.Dropout(dropout_rate)
                self.dropout2 = keras.layers.Dropout(dropout_rate)
            
            def call(self, inputs, encoder_output=None, training=None):
                # 自注意力
                attn1 = self.self_attention(inputs, inputs, training=training)
                attn1 = self.dropout1(attn1, training=training)
                out1 = self.layernorm1(inputs + attn1)
                
                # 交叉注意力（如果提供了encoder输出）
                if encoder_output is not None:
                    attn2 = self.cross_attention(out1, encoder_output, training=training)
                    attn2 = self.dropout2(attn2, training=training)
                    out2 = self.layernorm2(out1 + attn2)
                else:
                    out2 = out1
                
                # 前馈网络
                ffn_output = self.ffn(out2, training=training)
                return self.layernorm3(out2 + ffn_output)
            
            def get_config(self):
                config = super().get_config()
                config.update({
                    "num_heads": self.num_heads,
                    "key_dim": self.key_dim,
                    "ff_dim": self.ff_dim,
                    "dropout_rate": self.dropout_rate,
                    "activation": self.activation
                })
                return config
        
        return TransformerDecoderBlock(
            num_heads=num_heads,
            key_dim=key_dim,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate,
            activation=activation,
            name=layer_name
        )


@register_layer
class PositionalEncodingLayer(LayerNode):
    """
    位置编码层
    
    为输入添加正弦/余弦位置编码，用于Transformer
    """
    layer_type = "positional_encoding"
    display_name = "PositionalEncoding"
    category = LayerCategory.ATTENTION
    description = "位置编码层，为序列添加位置信息"
    icon = "📍"
    keras_class = "PositionalEncoding"
    
    def _setup_parameters(self):
        self.add_parameter(
            "max_length", "int", 512,
            display_name="最大序列长度",
            description="支持的最大序列长度",
            min_value=1,
            max_value=10000
        )
        self.add_parameter(
            "encoding_type", "choice", "sinusoidal",
            display_name="编码类型",
            description="位置编码的类型",
            choices=["sinusoidal", "learned"]
        )
        self.add_parameter(
            "dropout_rate", "float", 0.1,
            display_name="Dropout率",
            description="添加位置编码后的Dropout比率",
            min_value=0.0,
            max_value=1.0
        )
    
    def build_keras_layer(self):
        from tensorflow import keras
        import tensorflow as tf
        import numpy as np
        
        max_length = self.get_parameter("max_length")
        encoding_type = self.get_parameter("encoding_type")
        dropout_rate = self.get_parameter("dropout_rate")
        layer_name = self.layer_name or "positional_encoding"
        
        class PositionalEncodingBlock(keras.layers.Layer):
            def __init__(self, max_length, encoding_type, dropout_rate, **kwargs):
                super().__init__(**kwargs)
                self.max_length = max_length
                self.encoding_type = encoding_type
                self.dropout_rate = dropout_rate
                self.dropout = keras.layers.Dropout(dropout_rate)
                self.pos_encoding = None
                self.pos_embedding = None
            
            def build(self, input_shape):
                d_model = input_shape[-1]
                
                if self.encoding_type == "sinusoidal":
                    # 正弦/余弦位置编码
                    positions = np.arange(self.max_length)[:, np.newaxis]
                    dims = np.arange(d_model)[np.newaxis, :]
                    angles = positions / np.power(10000, (2 * (dims // 2)) / d_model)
                    
                    # 偶数维度使用sin，奇数维度使用cos
                    angles[:, 0::2] = np.sin(angles[:, 0::2])
                    angles[:, 1::2] = np.cos(angles[:, 1::2])
                    
                    self.pos_encoding = tf.constant(
                        angles[np.newaxis, :, :].astype(np.float32)
                    )
                else:
                    # 可学习的位置嵌入
                    self.pos_embedding = self.add_weight(
                        name="pos_embedding",
                        shape=(1, self.max_length, d_model),
                        initializer="random_normal",
                        trainable=True
                    )
                
                super().build(input_shape)
            
            def call(self, inputs, training=None):
                seq_len = tf.shape(inputs)[1]
                
                if self.encoding_type == "sinusoidal":
                    pos_enc = self.pos_encoding[:, :seq_len, :]
                else:
                    pos_enc = self.pos_embedding[:, :seq_len, :]
                
                output = inputs + pos_enc
                return self.dropout(output, training=training)
            
            def get_config(self):
                config = super().get_config()
                config.update({
                    "max_length": self.max_length,
                    "encoding_type": self.encoding_type,
                    "dropout_rate": self.dropout_rate
                })
                return config
        
        return PositionalEncodingBlock(
            max_length=max_length,
            encoding_type=encoding_type,
            dropout_rate=dropout_rate,
            name=layer_name
        )


@register_layer
class AdditiveAttentionLayer(LayerNode):
    """加性注意力层（Bahdanau Attention）"""
    layer_type = "additive_attention"
    display_name = "AdditiveAttention"
    category = LayerCategory.ATTENTION
    description = "加性注意力机制（Bahdanau风格）"
    icon = "➕"
    keras_class = "AdditiveAttention"
    
    def _setup_parameters(self):
        self.add_parameter(
            "use_scale", "bool", True,
            display_name="使用缩放",
            description="是否对注意力分数进行缩放"
        )
        self.add_parameter(
            "dropout", "float", 0.0,
            display_name="Dropout率",
            description="注意力权重的Dropout比率",
            min_value=0.0,
            max_value=1.0
        )
    
    def build_keras_layer(self):
        from tensorflow import keras
        
        return keras.layers.AdditiveAttention(
            use_scale=self.get_parameter("use_scale"),
            dropout=self.get_parameter("dropout"),
            name=self.layer_name or None
        )


@register_layer
class AttentionLayer(LayerNode):
    """点积注意力层（Luong Attention）"""
    layer_type = "attention"
    display_name = "Attention"
    category = LayerCategory.ATTENTION
    description = "点积注意力机制（Luong风格）"
    icon = "·"
    keras_class = "Attention"
    
    def _setup_parameters(self):
        self.add_parameter(
            "use_scale", "bool", False,
            display_name="使用缩放",
            description="是否对注意力分数进行缩放"
        )
        self.add_parameter(
            "score_mode", "choice", "dot",
            display_name="评分模式",
            description="计算注意力分数的方式",
            choices=["dot", "concat"]
        )
        self.add_parameter(
            "dropout", "float", 0.0,
            display_name="Dropout率",
            description="注意力权重的Dropout比率",
            min_value=0.0,
            max_value=1.0
        )
    
    def build_keras_layer(self):
        from tensorflow import keras
        
        return keras.layers.Attention(
            use_scale=self.get_parameter("use_scale"),
            score_mode=self.get_parameter("score_mode"),
            dropout=self.get_parameter("dropout"),
            name=self.layer_name or None
        )

