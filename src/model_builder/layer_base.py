# -*- coding: utf-8 -*-
"""
Layer Node Base Class Definition

Defines the base structure and interface for neural network layer nodes.

Architecture Notes
------------------
- Parameter definitions uniformly use the `Parameter` class from `src/core/parameter.py`
- `LayerParameter` is an alias for `Parameter`, maintaining backward compatibility
- This class is independent from `GraphNodeBase` in `src/core/graph_base.py`,
  because model layers need Keras layer building, shape inference, and other specific features
"""

import logging
import uuid
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type, Union

# 从核心模块导入统一的参数类
from src.core.parameter import Parameter as LayerParameter

logger = logging.getLogger(__name__)


class LayerCategory(Enum):
    """层分类"""
    INPUT = "input"              # 输入层
    CORE = "core"                # 核心层 (Dense, Embedding等)
    CONV = "conv"                # 卷积层
    RECURRENT = "recurrent"      # 循环层 (LSTM, GRU等)
    ATTENTION = "attention"      # 注意力层
    POOLING = "pooling"          # 池化层
    NORMALIZATION = "norm"       # 归一化层
    REGULARIZATION = "reg"       # 正则化层 (Dropout等)
    RESHAPE = "reshape"          # 形状变换层
    ACTIVATION = "activation"    # 激活函数层
    MERGE = "merge"              # 合并层
    OUTPUT = "output"            # 输出层
    
    @property
    def display_name(self) -> str:
        """获取显示名称"""
        names = {
            LayerCategory.INPUT: "输入",
            LayerCategory.CORE: "核心层",
            LayerCategory.CONV: "卷积层",
            LayerCategory.RECURRENT: "循环层",
            LayerCategory.ATTENTION: "注意力层",
            LayerCategory.POOLING: "池化层",
            LayerCategory.NORMALIZATION: "归一化",
            LayerCategory.REGULARIZATION: "正则化",
            LayerCategory.RESHAPE: "形状变换",
            LayerCategory.ACTIVATION: "激活函数",
            LayerCategory.MERGE: "合并层",
            LayerCategory.OUTPUT: "输出",
        }
        return names.get(self, self.value)
    
    @property
    def color(self) -> str:
        """获取分类颜色（Catppuccin Mocha）"""
        colors = {
            LayerCategory.INPUT: "#89b4fa",       # 蓝色
            LayerCategory.CORE: "#a6e3a1",        # 绿色
            LayerCategory.CONV: "#f9e2af",        # 黄色
            LayerCategory.RECURRENT: "#cba6f7",   # 紫色
            LayerCategory.ATTENTION: "#f38ba8",   # 红色
            LayerCategory.POOLING: "#94e2d5",     # 青色
            LayerCategory.NORMALIZATION: "#fab387",  # 橙色
            LayerCategory.REGULARIZATION: "#f5c2e7", # 粉色
            LayerCategory.RESHAPE: "#74c7ec",     # 天蓝
            LayerCategory.ACTIVATION: "#b4befe",  # 淡紫
            LayerCategory.MERGE: "#eba0ac",       # 玫红
            LayerCategory.OUTPUT: "#a6e3a1",      # 绿色
        }
        return colors.get(self, "#cdd6f4")


class LayerNode(ABC):
    """
    层节点基类
    
    所有神经网络层必须继承此类。
    """
    
    # 类属性 - 子类需要覆盖
    layer_type: str = "base_layer"        # 层类型标识
    display_name: str = "Base Layer"      # 显示名称
    category: LayerCategory = LayerCategory.CORE
    description: str = ""                 # 层描述
    icon: str = "🔲"                       # 层图标
    keras_class: str = ""                 # 对应的Keras类名
    
    def __init__(self, layer_id: str = None):
        """初始化层节点"""
        self.layer_id = layer_id or str(uuid.uuid4())[:8]
        self.parameters: Dict[str, LayerParameter] = {}
        self.parameter_values: Dict[str, Any] = {}
        
        # 位置信息（用于UI）
        self.position: Tuple[float, float] = (0.0, 0.0)
        
        # 输入输出形状（用于验证）
        self.input_shape: Optional[Tuple] = None
        self.output_shape: Optional[Tuple] = None
        
        # 层名称（用于Keras）
        self.layer_name: str = ""
        
        # 运行时状态（用于导入的模型）
        self._has_weights: bool = False    # 是否有权重
        self._trainable: bool = True       # 是否可训练（未冻结）
        
        # 初始化参数
        self._setup_parameters()
        self._init_parameter_values()
    
    @property
    def has_weights(self) -> bool:
        """是否有权重"""
        return self._has_weights
    
    @property
    def trainable(self) -> bool:
        """是否可训练（未冻结）"""
        return self._trainable
    
    @trainable.setter
    def trainable(self, value: bool):
        """设置可训练状态"""
        self._trainable = value
    
    @abstractmethod
    def _setup_parameters(self):
        """
        设置层参数
        
        子类必须实现此方法，使用 add_parameter 添加参数。
        """
        pass
    
    def _init_parameter_values(self):
        """初始化参数值为默认值"""
        for name, param in self.parameters.items():
            self.parameter_values[name] = param.default_value
    
    def add_parameter(
        self,
        name: str,
        param_type: str,
        default_value: Any,
        display_name: str = None,
        description: str = "",
        min_value: Any = None,
        max_value: Any = None,
        choices: List[Any] = None,
        required: bool = True
    ):
        """添加层参数"""
        param = LayerParameter(
            name=name,
            display_name=display_name or name,
            param_type=param_type,
            default_value=default_value,
            description=description,
            min_value=min_value,
            max_value=max_value,
            choices=choices,
            required=required
        )
        self.parameters[name] = param
        self.parameter_values[name] = default_value
    
    def get_parameter(self, name: str) -> Any:
        """获取参数值"""
        return self.parameter_values.get(name)
    
    def set_parameter(self, name: str, value: Any) -> Tuple[bool, str]:
        """设置参数值"""
        if name not in self.parameters:
            return False, f"未知参数: {name}"
        
        param = self.parameters[name]
        valid, msg = param.validate(value)
        if not valid:
            return False, msg
        
        self.parameter_values[name] = value
        return True, ""
    
    @abstractmethod
    def build_keras_layer(self):
        """
        构建Keras层
        
        返回: Keras层实例
        """
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """获取层配置（用于Keras模型构建）"""
        config = {}
        for name, value in self.parameter_values.items():
            if value is not None:
                config[name] = value
        if self.layer_name:
            config['name'] = self.layer_name
        return config
    
    def validate(self) -> Tuple[bool, str]:
        """验证层配置"""
        for name, value in self.parameter_values.items():
            if name in self.parameters:
                valid, msg = self.parameters[name].validate(value)
                if not valid:
                    return False, msg
        return True, ""
    
    def compute_output_shape(self, input_shape: Tuple) -> Tuple:
        """计算输出形状（子类可覆盖）"""
        # 默认保持输入形状不变
        return input_shape
    
    def to_dict(self) -> Dict:
        """序列化层为字典"""
        data = {
            "id": self.layer_id,
            "type": self.layer_type,
            "position": list(self.position),
            "parameters": dict(self.parameter_values),
            "layer_name": self.layer_name,
        }
        # 添加冻结状态（仅当冻结时保存）
        if not self._trainable:
            data['trainable'] = False
        # 添加权重状态（仅当有权重时保存）
        if self._has_weights:
            data['has_weights'] = True
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'LayerNode':
        """从字典反序列化层"""
        layer = cls(layer_id=data.get("id"))
        layer.position = tuple(data.get("position", [0, 0]))
        layer.layer_name = data.get("layer_name", "")
        for name, value in data.get("parameters", {}).items():
            layer.set_parameter(name, value)
        # 恢复冻结状态
        layer._trainable = data.get('trainable', True)
        # 恢复权重状态
        layer._has_weights = data.get('has_weights', False)
        return layer
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(id={self.layer_id}, type={self.layer_type})>"


# 层注册表
_layer_registry: Dict[str, Type[LayerNode]] = {}


def register_layer(layer_class: Type[LayerNode]) -> Type[LayerNode]:
    """
    层注册装饰器
    
    使用方法:
        @register_layer
        class DenseLayer(LayerNode):
            layer_type = "dense"
            ...
    """
    _layer_registry[layer_class.layer_type] = layer_class
    logger.debug(f"注册层: {layer_class.layer_type} -> {layer_class.__name__}")
    return layer_class


def get_layer_class(layer_type: str) -> Optional[Type[LayerNode]]:
    """根据类型获取层类"""
    return _layer_registry.get(layer_type)


def get_all_layer_types() -> List[str]:
    """获取所有已注册的层类型"""
    return list(_layer_registry.keys())


def get_layers_by_category(category: LayerCategory) -> List[Type[LayerNode]]:
    """获取指定分类的所有层类"""
    return [
        cls for cls in _layer_registry.values()
        if cls.category == category
    ]


def create_layer(layer_type: str, layer_id: str = None) -> Optional[LayerNode]:
    """根据类型创建层实例"""
    layer_class = get_layer_class(layer_type)
    if layer_class:
        return layer_class(layer_id=layer_id)
    return None

