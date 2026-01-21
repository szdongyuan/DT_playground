# -*- coding: utf-8 -*-
"""
Shape Transformation Layer Nodes
"""

from ..layer_base import LayerCategory, LayerNode, register_layer


@register_layer
class FlattenLayer(LayerNode):
    """展平层"""
    layer_type = "flatten"
    display_name = "Flatten 展平"
    category = LayerCategory.RESHAPE
    description = "将输入展平为一维"
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
    display_name = "Reshape 重塑"
    category = LayerCategory.RESHAPE
    description = "将输入重塑为指定形状"
    icon = "🔄"
    keras_class = "Reshape"
    
    def _setup_parameters(self):
        self.add_parameter(
            "target_shape", "str", "(64, 1)",
            display_name="目标形状",
            description="目标形状（不包含批次维度）"
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
    display_name = "Permute 置换"
    category = LayerCategory.RESHAPE
    description = "置换输入的维度"
    icon = "🔀"
    keras_class = "Permute"
    
    def _setup_parameters(self):
        self.add_parameter(
            "dims", "str", "(2, 1)",
            display_name="维度顺序",
            description="置换模式，1-based索引"
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
    description = "将输入重复n次"
    icon = "🔁"
    keras_class = "RepeatVector"
    
    def _setup_parameters(self):
        self.add_parameter(
            "n", "int", 1,
            display_name="重复次数",
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
    description = "一维上采样层"
    icon = "⬆️"
    keras_class = "UpSampling1D"
    
    def _setup_parameters(self):
        self.add_parameter(
            "size", "int", 2,
            display_name="上采样因子",
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
    description = "二维上采样层"
    icon = "⬆️"
    keras_class = "UpSampling2D"
    
    def _setup_parameters(self):
        self.add_parameter(
            "size", "str", "(2, 2)",
            display_name="上采样因子"
        )
    
    def build_keras_layer(self):
        from tensorflow import keras
        
        size = eval(self.get_parameter("size"))
        return keras.layers.UpSampling2D(
            size=size,
            name=self.layer_name or None
        )

