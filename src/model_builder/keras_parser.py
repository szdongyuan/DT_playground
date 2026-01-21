# -*- coding: utf-8 -*-
"""
Keras Model Parser

Generates ModelGraph from trained Keras models for fine-tuning and architecture editing.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

logger = logging.getLogger(__name__)


# Keras 层类型 -> 平台层类型 映射表
KERAS_TO_PLATFORM_TYPE: Dict[str, str] = {
    # 输入/输出
    'InputLayer': 'input',
    
    # 核心层
    'Dense': 'dense',
    'Embedding': 'embedding',
    
    # 卷积层
    'Conv1D': 'conv1d',
    'Conv2D': 'conv2d',
    'Conv1DTranspose': 'conv1d_transpose',
    'Conv2DTranspose': 'conv2d_transpose',
    'SeparableConv1D': 'separable_conv1d',
    'SeparableConv2D': 'separable_conv2d',
    'DepthwiseConv2D': 'depthwise_conv2d',
    
    # 循环层
    'LSTM': 'lstm',
    'GRU': 'gru',
    'SimpleRNN': 'simple_rnn',
    'Bidirectional': 'bidirectional',
    
    # 池化层
    'MaxPooling1D': 'max_pooling1d',
    'MaxPooling2D': 'max_pooling2d',
    'AveragePooling1D': 'avg_pooling1d',
    'AveragePooling2D': 'avg_pooling2d',
    'GlobalMaxPooling1D': 'global_max_pooling1d',
    'GlobalMaxPooling2D': 'global_max_pooling2d',
    'GlobalAveragePooling1D': 'global_avg_pooling1d',
    'GlobalAveragePooling2D': 'global_avg_pooling2d',
    
    # 归一化层
    'BatchNormalization': 'batch_norm',
    'LayerNormalization': 'layer_norm',
    
    # 正则化层
    'Dropout': 'dropout',
    'SpatialDropout1D': 'spatial_dropout1d',
    'SpatialDropout2D': 'spatial_dropout2d',
    'GaussianNoise': 'gaussian_noise',
    'GaussianDropout': 'gaussian_dropout',
    
    # 形状变换层
    'Flatten': 'flatten',
    'Reshape': 'reshape',
    'Permute': 'permute',
    'RepeatVector': 'repeat_vector',
    'UpSampling1D': 'upsampling1d',
    'UpSampling2D': 'upsampling2d',
    
    # 激活层
    'Activation': 'activation',
    'LeakyReLU': 'leaky_relu',
    'PReLU': 'prelu',
    'ELU': 'elu',
    'Softmax': 'softmax',
    'ReLU': 'activation',  # 映射到通用激活层
    
    # 合并层
    'Concatenate': 'concatenate',
    'Add': 'add',
    'Multiply': 'multiply',
    'Average': 'average',
    'Maximum': 'maximum',
    'Minimum': 'minimum',
    'Dot': 'dot',
    
    # 注意力层
    'MultiHeadAttention': 'multi_head_attention',
    'Attention': 'attention',
    'AdditiveAttention': 'additive_attention',
}


@dataclass
class ParsedLayerInfo:
    """解析后的层信息"""
    name: str                       # Keras 层名称
    layer_type: str                 # 平台层类型
    keras_class_name: str           # 原始 Keras 类名
    config: Dict[str, Any]          # 层配置参数
    input_layer_names: List[str]    # 输入层名称列表
    has_weights: bool               # 是否有权重
    trainable: bool                 # 是否可训练
    position: Tuple[float, float] = (0.0, 0.0)  # 画布位置


class KerasModelParser:
    """
    Keras 模型解析器
    
    将已训练的 Keras 模型转换为平台的 ModelGraph 格式。
    """
    
    def __init__(self):
        self._unsupported_layers: List[str] = []
        self._keras_model = None
    
    def parse(self, keras_model) -> 'ModelGraph':
        """
        解析 Keras 模型
        
        Args:
            keras_model: tf.keras.Model 实例
            
        Returns:
            ModelGraph 实例
        """
        from .model_graph import ModelGraph, ModelConnection
        from .layer_base import create_layer, get_layer_class
        
        self._keras_model = keras_model
        self._unsupported_layers = []
        
        # 创建模型图
        model_name = keras_model.name or "imported_model"
        # 清理非法字符
        import re
        model_name = re.sub(r'[^A-Za-z0-9._/>-]', '_', model_name)
        
        graph = ModelGraph(name=model_name)
        
        # 解析所有层
        layer_infos = self._parse_all_layers(keras_model)
        
        # 计算自动布局位置
        self._calculate_layout(layer_infos)
        
        # 添加层到图中
        for info in layer_infos:
            layer_node = self._create_layer_node(info)
            if layer_node:
                layer_node.position = info.position
                layer_node.layer_id = info.name  # 使用 Keras 层名作为 ID
                graph.layers[layer_node.layer_id] = layer_node
                logger.debug(f"添加层: {info.name} ({info.layer_type})")
        
        # 添加连接（优先使用 inbound_nodes，否则使用顺序推断）
        connections_added = set()
        for info in layer_infos:
            for input_name in info.input_layer_names:
                if input_name in graph.layers and info.name in graph.layers:
                    conn_key = (input_name, info.name)
                    if conn_key not in connections_added:
                        graph.connections.append(ModelConnection(input_name, info.name))
                        connections_added.add(conn_key)
        
        # 如果没有通过 inbound_nodes 获取到连接，使用顺序推断（适用于 Sequential 模型）
        if not connections_added and len(layer_infos) > 1:
            logger.debug("使用顺序推断连接（Sequential 模型）")
            valid_layers = [info for info in layer_infos if info.name in graph.layers]
            for i in range(len(valid_layers) - 1):
                source_name = valid_layers[i].name
                target_name = valid_layers[i + 1].name
                graph.connections.append(ModelConnection(source_name, target_name))
        
        # 记录不支持的层
        if self._unsupported_layers:
            unique_unsupported = list(set(self._unsupported_layers))
            logger.warning(f"以下层类型暂不支持，已跳过: {unique_unsupported}")
        
        graph.is_dirty = False
        return graph
    
    def _parse_all_layers(self, keras_model) -> List[ParsedLayerInfo]:
        """解析模型中的所有层"""
        layer_infos = []
        
        for layer in keras_model.layers:
            info = self._parse_single_layer(layer)
            if info:
                layer_infos.append(info)
        
        return layer_infos
    
    def _parse_single_layer(self, keras_layer) -> Optional[ParsedLayerInfo]:
        """解析单个 Keras 层"""
        keras_class_name = keras_layer.__class__.__name__
        platform_type = KERAS_TO_PLATFORM_TYPE.get(keras_class_name)
        
        if platform_type is None:
            self._unsupported_layers.append(keras_class_name)
            logger.debug(f"未知层类型: {keras_class_name}, 将跳过")
            return None
        
        # 获取层配置
        config = self._extract_layer_config(keras_layer, platform_type)
        
        # 获取输入层名称
        input_layer_names = self._get_input_layer_names(keras_layer)
        
        # 检查权重
        has_weights = len(keras_layer.weights) > 0
        
        return ParsedLayerInfo(
            name=keras_layer.name,
            layer_type=platform_type,
            keras_class_name=keras_class_name,
            config=config,
            input_layer_names=input_layer_names,
            has_weights=has_weights,
            trainable=keras_layer.trainable
        )
    
    def _extract_layer_config(self, keras_layer, platform_type: str) -> Dict[str, Any]:
        """提取层配置参数"""
        config = {}
        keras_config = keras_layer.get_config()
        
        # 通用参数映射
        common_params = [
            'units', 'filters', 'kernel_size', 'strides', 'padding',
            'activation', 'use_bias', 'rate', 'momentum', 'epsilon',
            'axis', 'pool_size', 'return_sequences', 'num_heads', 'key_dim',
            'value_dim', 'dropout', 'alpha', 'stddev', 'size', 'dims',
            'target_shape', 'n', 'input_dim', 'output_dim', 'mask_zero',
        ]
        
        for param in common_params:
            if param in keras_config:
                value = keras_config[param]
                # 处理元组/列表转字符串（用于 UI 显示）
                if isinstance(value, (list, tuple)):
                    if len(value) == 1:
                        config[param] = value[0]
                    else:
                        config[param] = str(tuple(value))
                elif value is not None:
                    config[param] = value
        
        # 特殊处理 Input 层
        if platform_type == 'input':
            if 'batch_input_shape' in keras_config:
                shape = keras_config['batch_input_shape'][1:]  # 去掉 batch 维度
                config['shape'] = str(tuple(shape)).replace('None', '')
            elif hasattr(keras_layer, 'input_shape') and keras_layer.input_shape:
                shape = keras_layer.input_shape[0][1:] if isinstance(keras_layer.input_shape, list) else keras_layer.input_shape[1:]
                config['shape'] = str(tuple(shape)).replace('None', '')
        
        # 处理 ReLU 映射到 activation
        if keras_layer.__class__.__name__ == 'ReLU':
            config['activation'] = 'relu'
        
        return config
    
    def _get_input_layer_names(self, keras_layer) -> List[str]:
        """获取层的输入层名称"""
        input_names = []
        
        # 使用 _inbound_nodes 获取输入连接
        if hasattr(keras_layer, '_inbound_nodes'):
            for node in keras_layer._inbound_nodes:
                if hasattr(node, 'inbound_layers'):
                    inbound_layers = node.inbound_layers
                    # 处理可能是单个层或列表的情况
                    if not isinstance(inbound_layers, (list, tuple)):
                        inbound_layers = [inbound_layers]
                    for inbound_layer in inbound_layers:
                        if hasattr(inbound_layer, 'name'):
                            input_names.append(inbound_layer.name)
        
        return input_names
    
    def _calculate_layout(self, layer_infos: List[ParsedLayerInfo]):
        """计算层的自动布局位置"""
        if not layer_infos:
            return
        
        # 简单的水平布局，每个层间隔 180 像素
        x_spacing = 180
        for i, info in enumerate(layer_infos):
            info.position = (i * x_spacing, 0.0)
    
    def _create_layer_node(self, info: ParsedLayerInfo) -> Optional['LayerNode']:
        """从 ParsedLayerInfo 创建 LayerNode"""
        from .layer_base import create_layer, get_layer_class
        
        # 检查平台是否支持该层类型
        layer_class = get_layer_class(info.layer_type)
        if layer_class is None:
            logger.warning(f"平台不支持层类型: {info.layer_type} (Keras: {info.keras_class_name})")
            self._unsupported_layers.append(info.keras_class_name)
            return None
        
        layer_node = create_layer(info.layer_type)
        if layer_node is None:
            logger.warning(f"无法创建层实例: {info.layer_type}")
            return None
        
        # 设置参数
        for param_name, value in info.config.items():
            if param_name in layer_node.parameters:
                layer_node.set_parameter(param_name, value)
        
        # 设置权重和训练状态
        layer_node._has_weights = info.has_weights
        layer_node._trainable = info.trainable
        layer_node.layer_name = info.name
        
        return layer_node
    
    def get_unsupported_layers(self) -> List[str]:
        """获取不支持的层类型列表"""
        return list(set(self._unsupported_layers))
    
    def get_model_summary(self, keras_model) -> Dict[str, Any]:
        """
        获取模型摘要信息
        
        Returns:
            包含层数、参数量等信息的字典
        """
        total_params = 0
        trainable_params = 0
        non_trainable_params = 0
        
        for weight in keras_model.trainable_weights:
            trainable_params += weight.numpy().size
        
        for weight in keras_model.non_trainable_weights:
            non_trainable_params += weight.numpy().size
        
        total_params = trainable_params + non_trainable_params
        
        return {
            'name': keras_model.name,
            'layer_count': len(keras_model.layers),
            'total_params': total_params,
            'trainable_params': trainable_params,
            'non_trainable_params': non_trainable_params,
        }


def parse_keras_model(model_path: str) -> Tuple['ModelGraph', Dict[str, Any]]:
    """
    便捷函数：从文件加载并解析 Keras 模型
    
    Args:
        model_path: .keras 或 .h5 文件路径
        
    Returns:
        (ModelGraph, 模型摘要信息)
    """
    import tensorflow as tf
    
    # 加载模型（不编译，避免优化器问题）
    keras_model = tf.keras.models.load_model(model_path, compile=False)
    
    # 解析
    parser = KerasModelParser()
    graph = parser.parse(keras_model)
    summary = parser.get_model_summary(keras_model)
    summary['unsupported_layers'] = parser.get_unsupported_layers()
    
    return graph, summary

