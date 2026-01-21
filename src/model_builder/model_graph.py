# -*- coding: utf-8 -*-
"""
Model Graph Data Structure

Manages neural network layer connections and topology.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .layer_base import LayerNode, create_layer, get_layer_class

logger = logging.getLogger(__name__)


@dataclass
class ModelConnection:
    """层连接定义"""
    source_layer_id: str      # 源层ID
    target_layer_id: str      # 目标层ID
    
    def to_dict(self) -> Dict:
        return {
            "source": self.source_layer_id,
            "target": self.target_layer_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelConnection':
        return cls(
            source_layer_id=data["source"],
            target_layer_id=data["target"],
        )


@dataclass
class ModelMetadata:
    """模型元数据"""
    name: str = "未命名模型"
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    modified_at: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = "1.0"


@dataclass
class CompileConfig:
    """模型编译配置"""
    optimizer: str = "Adam"
    learning_rate: float = 0.001
    loss: str = "mse"
    metrics: List[str] = field(default_factory=lambda: ["mae"])
    
    # 可用选项
    OPTIMIZERS = ["Adam", "SGD", "RMSprop", "AdamW", "Nadam"]
    LOSSES = [
        "mse", "mae", "huber",  # 回归
        "binary_crossentropy", "categorical_crossentropy", "sparse_categorical_crossentropy",  # 分类
    ]
    METRICS = [
        "accuracy", "mae", "mse", "rmse",
        "precision", "recall", "auc",
        "cosine_similarity",
    ]
    
    def to_dict(self) -> Dict:
        return {
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "loss": self.loss,
            "metrics": self.metrics,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CompileConfig':
        return cls(
            optimizer=data.get("optimizer", "Adam"),
            learning_rate=data.get("learning_rate", 0.001),
            loss=data.get("loss", "mse"),
            metrics=data.get("metrics", ["mae"]),
        )
    
    @staticmethod
    def create_optimizer(optimizer_name: str, learning_rate: float = 0.001):
        """
        创建 Keras 优化器实例
        
        Args:
            optimizer_name: 优化器名称（不区分大小写）
            learning_rate: 学习率
            
        Returns:
            Keras 优化器实例
        """
        from tensorflow import keras
        
        optimizer_map = {
            'adam': keras.optimizers.Adam,
            'sgd': keras.optimizers.SGD,
            'rmsprop': keras.optimizers.RMSprop,
            'adamw': keras.optimizers.AdamW,
            'nadam': keras.optimizers.Nadam,
        }
        
        optimizer_cls = optimizer_map.get(optimizer_name.lower(), keras.optimizers.Adam)
        return optimizer_cls(learning_rate=learning_rate)
    
    @staticmethod
    def auto_detect_loss(model_or_labels) -> str:
        """
        自动检测合适的损失函数
        
        Args:
            model_or_labels: Keras 模型实例或标签数据 (numpy array)
            
        Returns:
            损失函数名称字符串
        """
        import numpy as np
        
        # 如果是 Keras 模型，根据输出层检测
        try:
            from tensorflow import keras
            if isinstance(model_or_labels, keras.Model):
                return CompileConfig._detect_loss_from_model(model_or_labels)
        except ImportError:
            pass
        
        # 如果是标签数据，根据数据特征检测
        if isinstance(model_or_labels, np.ndarray):
            return CompileConfig._detect_loss_from_labels(model_or_labels)
        
        # 默认返回 mse
        return "mse"
    
    @staticmethod
    def _detect_loss_from_model(model) -> str:
        """从模型输出层检测损失函数"""
        output_shape = model.output_shape
        if isinstance(output_shape, list):
            output_shape = output_shape[0]
        
        output_dim = output_shape[-1]
        
        # 检测最后一层激活函数
        last_layer = model.layers[-1]
        if hasattr(last_layer, 'activation'):
            activation = last_layer.activation.__name__
            if activation == 'sigmoid':
                return 'binary_crossentropy'
            elif activation == 'softmax':
                return 'categorical_crossentropy'
            elif activation == 'linear':
                return 'mse'
        
        # 根据输出维度判断
        if output_dim == 1:
            return 'binary_crossentropy'
        elif output_dim > 1:
            return 'categorical_crossentropy'
        
        return 'mse'
    
    @staticmethod
    def _detect_loss_from_labels(y) -> str:
        """从标签数据检测损失函数"""
        import numpy as np
        
        # 如果 y 是多维数组（如音频波形、频谱图），通常是回归任务（自编码器）
        if len(y.shape) >= 2 and y.shape[-1] != 1:
            # 检查是否是连续值（回归/自编码器）
            if y.dtype in [np.float32, np.float64]:
                # 如果值范围较大或有负值，可能是波形数据，使用 MSE
                if y.min() < 0 or y.max() > 10:
                    return "mse"
                # 检查是否是 one-hot 编码的分类标签
                if np.allclose(y.sum(axis=-1), 1.0) and np.all((y >= 0) & (y <= 1)):
                    return "categorical_crossentropy"
            return "mse"
        
        # 一维标签
        if len(y.shape) == 1:
            try:
                y_flat = y.flatten()
                if np.issubdtype(y.dtype, np.number):
                    unique_values = len(np.unique(y_flat))
                    if unique_values == 2:
                        return "binary_crossentropy"
                    else:
                        return "sparse_categorical_crossentropy"
            except Exception:
                pass
            return "mse"
        
        return "mse"
    
    @staticmethod
    def get_default_metrics(loss: str) -> List[str]:
        """
        根据损失函数获取默认评估指标
        
        Args:
            loss: 损失函数名称
            
        Returns:
            评估指标列表
        """
        if 'crossentropy' in loss:
            return ['accuracy']
        return ['mae']


class ModelGraph:
    """
    模型图
    
    管理神经网络的层和连接关系。
    """
    
    def __init__(self, name: str = "未命名模型"):
        self.metadata = ModelMetadata(name=name)
        self.layers: Dict[str, LayerNode] = {}
        self.connections: List[ModelConnection] = []
        self.compile_config: CompileConfig = CompileConfig()
        self.is_dirty: bool = False
    
    @property
    def name(self) -> str:
        return self.metadata.name
    
    @name.setter
    def name(self, value: str):
        self.metadata.name = value
        self._mark_dirty()
    
    def add_layer(self, layer: LayerNode) -> str:
        """添加层"""
        self.layers[layer.layer_id] = layer
        self._mark_dirty()
        return layer.layer_id
    
    def remove_layer(self, layer_id: str) -> bool:
        """移除层及其连接"""
        if layer_id not in self.layers:
            return False
        
        # 移除相关连接
        self.connections = [
            conn for conn in self.connections
            if conn.source_layer_id != layer_id and conn.target_layer_id != layer_id
        ]
        
        del self.layers[layer_id]
        self._mark_dirty()
        return True
    
    def get_layer(self, layer_id: str) -> Optional[LayerNode]:
        """获取层"""
        return self.layers.get(layer_id)
    
    def create_and_add_layer(self, layer_type: str, position: Tuple[float, float] = None) -> Optional[LayerNode]:
        """创建并添加层"""
        layer = create_layer(layer_type)
        if not layer:
            logger.error(f"无法创建层类型: {layer_type}")
            return None
        
        if position:
            layer.position = position
        
        self.add_layer(layer)
        return layer
    
    def connect(self, source_layer_id: str, target_layer_id: str) -> Tuple[bool, str]:
        """连接两个层"""
        # 检查层是否存在
        if source_layer_id not in self.layers:
            return False, f"源层不存在: {source_layer_id}"
        if target_layer_id not in self.layers:
            return False, f"目标层不存在: {target_layer_id}"
        
        # 检查是否自连接
        if source_layer_id == target_layer_id:
            return False, "不能自连接"
        
        # 检查是否已存在连接
        for conn in self.connections:
            if conn.source_layer_id == source_layer_id and conn.target_layer_id == target_layer_id:
                return False, "连接已存在"
        
        # 检查是否会创建循环（简单检测）
        if self._would_create_cycle(source_layer_id, target_layer_id):
            return False, "不能创建循环连接"
        
        # 添加连接
        connection = ModelConnection(
            source_layer_id=source_layer_id,
            target_layer_id=target_layer_id
        )
        self.connections.append(connection)
        self._mark_dirty()
        return True, ""
    
    def disconnect(self, source_layer_id: str, target_layer_id: str) -> bool:
        """断开连接"""
        for i, conn in enumerate(self.connections):
            if conn.source_layer_id == source_layer_id and conn.target_layer_id == target_layer_id:
                self.connections.pop(i)
                self._mark_dirty()
                return True
        return False
    
    def disconnect_layer(self, layer_id: str) -> bool:
        """
        移除指定层的所有连接
        
        Args:
            layer_id: 层ID
        
        Returns:
            是否有连接被移除
        """
        if layer_id not in self.layers:
            return False
        
        # 收集该层相关的所有连接
        connections_to_remove = [
            conn for conn in self.connections
            if conn.source_layer_id == layer_id or conn.target_layer_id == layer_id
        ]
        
        if not connections_to_remove:
            return False
        
        # 移除连接
        for conn in connections_to_remove:
            self.connections.remove(conn)
        
        self._mark_dirty()
        logger.debug(f"移除层 {layer_id} 的所有连接: {len(connections_to_remove)} 个")
        return True
    
    def _would_create_cycle(self, source: str, target: str) -> bool:
        """检查添加连接是否会创建循环"""
        # 从target开始，看能否到达source
        visited = set()
        stack = [target]
        
        while stack:
            current = stack.pop()
            if current == source:
                return True
            if current in visited:
                continue
            visited.add(current)
            
            # 找到从current出发的所有连接
            for conn in self.connections:
                if conn.source_layer_id == current:
                    stack.append(conn.target_layer_id)
        
        return False
    
    def get_execution_order(self) -> List[str]:
        """
        获取拓扑排序的执行顺序
        
        Returns:
            按执行顺序排列的层ID列表
        """
        # 计算入度
        in_degree = {layer_id: 0 for layer_id in self.layers}
        for conn in self.connections:
            in_degree[conn.target_layer_id] += 1
        
        # 找到入度为0的节点
        queue = [lid for lid, deg in in_degree.items() if deg == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            for conn in self.connections:
                if conn.source_layer_id == current:
                    in_degree[conn.target_layer_id] -= 1
                    if in_degree[conn.target_layer_id] == 0:
                        queue.append(conn.target_layer_id)
        
        return result
    
    def get_input_layers(self) -> List[str]:
        """获取输入层（没有输入连接的层）"""
        has_input = set(conn.target_layer_id for conn in self.connections)
        return [lid for lid in self.layers if lid not in has_input]
    
    def get_output_layers(self) -> List[str]:
        """获取输出层（没有输出连接的层）"""
        has_output = set(conn.source_layer_id for conn in self.connections)
        return [lid for lid in self.layers if lid not in has_output]
    
    def get_upstream_layers(self, layer_id: str) -> List[str]:
        """获取指定层的上游层"""
        return [
            conn.source_layer_id
            for conn in self.connections
            if conn.target_layer_id == layer_id
        ]
    
    def get_downstream_layers(self, layer_id: str) -> List[str]:
        """获取指定层的下游层"""
        return [
            conn.target_layer_id
            for conn in self.connections
            if conn.source_layer_id == layer_id
        ]
    
    def validate(self) -> Tuple[bool, List[str]]:
        """验证模型图"""
        errors = []
        
        if not self.layers:
            errors.append("模型没有任何层")
        
        # 检查是否有输入层
        input_layers = self.get_input_layers()
        if not input_layers:
            errors.append("模型没有输入层")
        
        # 检查是否有输出层
        output_layers = self.get_output_layers()
        if not output_layers:
            errors.append("模型没有输出层")
        
        # 验证每个层
        for layer_id, layer in self.layers.items():
            valid, msg = layer.validate()
            if not valid:
                errors.append(f"层 {layer.display_name}: {msg}")
        
        # 检查是否所有层都已连接
        order = self.get_execution_order()
        if len(order) != len(self.layers):
            errors.append("存在循环连接或断开的层")
        
        return len(errors) == 0, errors
    
    def build_keras_model(self, compile_model: bool = True):
        """
        构建Keras模型
        
        Args:
            compile_model: 是否编译模型（应用优化器、损失函数和指标）
        
        Returns:
            Keras Model实例
        """
        import tensorflow as tf
        from tensorflow import keras
        
        # 获取执行顺序
        order = self.get_execution_order()
        if not order:
            raise ValueError("无法确定层的执行顺序")
        
        # 存储每层的输出张量
        layer_outputs: Dict[str, Any] = {}
        model_inputs = []
        
        for layer_id in order:
            layer = self.layers[layer_id]
            upstream = self.get_upstream_layers(layer_id)
            
            if not upstream:
                # 这是输入层（没有上游连接）
                if layer.layer_type == 'input':
                    # 是 InputLayer 类型
                    keras_tensor = layer.build_keras_layer()  # Input() 返回张量
                    layer_outputs[layer_id] = keras_tensor
                    model_inputs.append(keras_tensor)
                else:
                    # 其他层作为入口，需要创建 Input 层
                    # 尝试从层参数获取形状 (shape 或 input_shape)
                    input_shape = layer.get_parameter('shape') or layer.get_parameter('input_shape')
                    if input_shape:
                        if isinstance(input_shape, str):
                            input_shape = eval(input_shape)
                        inp = keras.layers.Input(shape=input_shape)
                        keras_layer = layer.build_keras_layer()
                        layer_outputs[layer_id] = keras_layer(inp)
                        model_inputs.append(inp)
                    else:
                        raise ValueError(f"层 {layer.display_name} 需要指定 shape 或 input_shape 参数")
            elif len(upstream) == 1:
                # 单输入
                prev_output = layer_outputs[upstream[0]]
                keras_layer = layer.build_keras_layer()
                layer_outputs[layer_id] = keras_layer(prev_output)
            else:
                # 多输入（合并层）
                prev_outputs = [layer_outputs[uid] for uid in upstream]
                keras_layer = layer.build_keras_layer()
                layer_outputs[layer_id] = keras_layer(prev_outputs)
        
        # 获取输出
        output_layer_ids = self.get_output_layers()
        if len(output_layer_ids) == 1:
            model_output = layer_outputs[output_layer_ids[0]]
        else:
            model_output = [layer_outputs[lid] for lid in output_layer_ids]
        
        # 创建模型
        if len(model_inputs) == 1:
            model = keras.Model(inputs=model_inputs[0], outputs=model_output, name=self.name)
        else:
            model = keras.Model(inputs=model_inputs, outputs=model_output, name=self.name)
        
        # 应用冻结状态
        self._apply_freeze_status(model)
        
        # 编译模型
        if compile_model:
            self._compile_model(model)
        
        return model
    
    def get_compile_config_from_output(self) -> Optional[Dict]:
        """
        从输出层获取编译配置
        
        Returns:
            编译配置字典，如果没有输出层则返回None
        """
        output_layer_ids = self.get_output_layers()
        for layer_id in output_layer_ids:
            layer = self.layers.get(layer_id)
            if layer and layer.layer_type == 'output':
                # 优先使用 Output 层的编译配置
                return layer.get_compile_config()
        
        # 如果没有 Output 层，使用默认的 compile_config
        return self.compile_config.to_dict()
    
    def _apply_freeze_status(self, model):
        """
        应用冻结状态到 Keras 模型
        
        将 ModelGraph 中各层的 trainable 状态应用到构建的 Keras 模型。
        """
        frozen_count = 0
        for layer_id, layer_node in self.layers.items():
            if not layer_node.trainable:
                # 使用 layer_name 或 layer_id 查找 Keras 层
                keras_layer_name = layer_node.layer_name or layer_id
                try:
                    keras_layer = model.get_layer(keras_layer_name)
                    keras_layer.trainable = False
                    frozen_count += 1
                    logger.debug(f"层已冻结: {keras_layer_name}")
                except ValueError:
                    # 层名不匹配，尝试用 layer_id
                    try:
                        keras_layer = model.get_layer(layer_id)
                        keras_layer.trainable = False
                        frozen_count += 1
                        logger.debug(f"层已冻结: {layer_id}")
                    except ValueError:
                        logger.warning(f"无法找到层进行冻结: {keras_layer_name} / {layer_id}")
        
        if frozen_count > 0:
            logger.info(f"已冻结 {frozen_count} 个层")
    
    def _compile_model(self, model):
        """编译模型"""
        # 优先从输出层获取编译配置
        config = self.get_compile_config_from_output()
        if config is None:
            config = self.compile_config.to_dict()
        
        optimizer_name = config.get('optimizer', 'Adam')
        learning_rate = config.get('learning_rate', 0.001)
        loss = config.get('loss', 'mse')
        metrics = config.get('metrics', ['mae'])
        
        # 使用统一的工厂方法创建优化器
        optimizer = CompileConfig.create_optimizer(optimizer_name, learning_rate)
        
        # 编译
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        logger.info(f"模型已编译: optimizer={optimizer_name}, "
                   f"loss={loss}, metrics={metrics}")
    
    def _mark_dirty(self):
        """标记为已修改"""
        self.is_dirty = True
        self.metadata.modified_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """序列化模型图"""
        return {
            "version": self.metadata.version,
            "name": self.metadata.name,
            "description": self.metadata.description,
            "created_at": self.metadata.created_at,
            "modified_at": self.metadata.modified_at,
            "compile_config": self.compile_config.to_dict(),
            "layers": [layer.to_dict() for layer in self.layers.values()],
            "connections": [conn.to_dict() for conn in self.connections],
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelGraph':
        """从字典反序列化模型图"""
        graph = cls(name=data.get("name", "未命名模型"))
        graph.metadata.description = data.get("description", "")
        graph.metadata.created_at = data.get("created_at", "")
        graph.metadata.modified_at = data.get("modified_at", "")
        graph.metadata.version = data.get("version", "1.0")
        
        # 加载编译配置
        if "compile_config" in data:
            graph.compile_config = CompileConfig.from_dict(data["compile_config"])
        
        # 加载层
        for layer_data in data.get("layers", []):
            layer_type = layer_data.get("type")
            layer_class = get_layer_class(layer_type)
            if layer_class:
                layer = layer_class.from_dict(layer_data)
                graph.layers[layer.layer_id] = layer
            else:
                logger.warning(f"未知层类型: {layer_type}")
        
        # 加载连接
        for conn_data in data.get("connections", []):
            graph.connections.append(ModelConnection.from_dict(conn_data))
        
        graph.is_dirty = False
        return graph
    
    def save(self, filepath: str):
        """保存模型图到文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        self.is_dirty = False
        logger.info(f"模型图已保存: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'ModelGraph':
        """从文件加载模型图"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        graph = cls.from_dict(data)
        logger.info(f"模型图已加载: {filepath}")
        return graph

