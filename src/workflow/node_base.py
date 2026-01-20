# -*- coding: utf-8 -*-
"""
节点基类定义

所有工作流节点的基类，定义节点的基本结构和接口。

注意: 此模块保持原有 API 以兼容现有节点实现。
核心抽象类定义在 src/core/graph_base.py 中。
未来可以考虑让 BaseNode 继承 GraphNodeBase。
"""

import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type

from .port import create_input_port, create_output_port, DataType, Port

# 引入核心参数类型（用于类型提示和未来迁移）
# from src.core.parameter import Parameter as CoreParameter, ParamType


logger = logging.getLogger(__name__)


class NodeCategory(Enum):
    """节点分类"""
    DATA_SOURCE = "data_source"       # 数据源
    PREPROCESSING = "preprocessing"   # 预处理
    AUGMENTATION = "augmentation"     # 数据增强
    FEATURE = "feature"               # 特征提取
    TRAINING = "training"             # 训练
    CONTROL = "control"               # 控制流
    OUTPUT = "output"                 # 输出
    
    @property
    def display_name(self) -> str:
        """获取显示名称"""
        names = {
            NodeCategory.DATA_SOURCE: "数据源",
            NodeCategory.PREPROCESSING: "预处理",
            NodeCategory.AUGMENTATION: "数据增强",
            NodeCategory.FEATURE: "特征提取",
            NodeCategory.TRAINING: "人工智能",
            NodeCategory.CONTROL: "控制流",
            NodeCategory.OUTPUT: "输出",
        }
        return names.get(self, self.value)
    
    @property
    def color(self) -> str:
        """获取分类颜色（Catppuccin Mocha）"""
        colors = {
            NodeCategory.DATA_SOURCE: "#89b4fa",    # 蓝色
            NodeCategory.PREPROCESSING: "#a6e3a1",  # 绿色
            NodeCategory.AUGMENTATION: "#f9e2af",   # 黄色
            NodeCategory.FEATURE: "#cba6f7",        # 紫色
            NodeCategory.TRAINING: "#f38ba8",       # 红色
            NodeCategory.CONTROL: "#94e2d5",        # 青色
            NodeCategory.OUTPUT: "#fab387",         # 橙色
        }
        return colors.get(self, "#cdd6f4")


@dataclass
class NodeParameter:
    """节点参数定义"""
    name: str                       # 参数名称
    display_name: str               # 显示名称
    param_type: str                 # 类型: int, float, str, bool, choice, file, folder
    default_value: Any              # 默认值
    description: str = ""           # 描述
    min_value: Any = None           # 最小值（数值类型）
    max_value: Any = None           # 最大值（数值类型）
    choices: List[Any] = None       # 选项列表（choice类型）
    file_filter: str = ""           # 文件过滤器（file类型）
    default_directory: str = ""     # 文件对话框默认目录（file/folder类型）
    
    def validate(self, value: Any) -> Tuple[bool, str]:
        """验证参数值"""
        if self.param_type == "int":
            if not isinstance(value, int):
                return False, f"参数 {self.display_name} 必须是整数"
            if self.min_value is not None and value < self.min_value:
                return False, f"参数 {self.display_name} 不能小于 {self.min_value}"
            if self.max_value is not None and value > self.max_value:
                return False, f"参数 {self.display_name} 不能大于 {self.max_value}"
        elif self.param_type == "float":
            if not isinstance(value, (int, float)):
                return False, f"参数 {self.display_name} 必须是数值"
            if self.min_value is not None and value < self.min_value:
                return False, f"参数 {self.display_name} 不能小于 {self.min_value}"
            if self.max_value is not None and value > self.max_value:
                return False, f"参数 {self.display_name} 不能大于 {self.max_value}"
        elif self.param_type == "choice":
            if self.choices and value not in self.choices:
                return False, f"参数 {self.display_name} 必须是 {self.choices} 之一"
        return True, ""


class NodeState(Enum):
    """节点执行状态"""
    IDLE = "idle"               # 空闲
    WAITING = "waiting"         # 等待输入
    RUNNING = "running"         # 执行中
    COMPLETED = "completed"     # 已完成
    ERROR = "error"             # 错误
    SKIPPED = "skipped"         # 跳过


class BaseNode(ABC):
    """
    节点基类
    
    所有工作流节点必须继承此类并实现 execute 方法。
    """
    
    # 类属性 - 子类需要覆盖
    node_type: str = "base_node"          # 节点类型标识
    display_name: str = "Base Node"       # 显示名称
    category: NodeCategory = NodeCategory.DATA_SOURCE
    description: str = ""                 # 节点描述
    icon: str = "📦"                       # 节点图标（emoji或图标路径）
    
    def __init__(self, node_id: str = None):
        """
        初始化节点
        
        Args:
            node_id: 节点唯一标识，默认自动生成
        """
        self.node_id = node_id or str(uuid.uuid4())[:8]
        self.inputs: Dict[str, Port] = {}
        self.outputs: Dict[str, Port] = {}
        self.parameters: Dict[str, NodeParameter] = {}
        self.parameter_values: Dict[str, Any] = {}
        
        # 节点状态
        self.state: NodeState = NodeState.IDLE
        self.error_message: str = ""
        
        # 位置信息（用于UI）
        self.position: Tuple[float, float] = (0.0, 0.0)
        
        # 进度回调函数: (progress: float, message: str, data: dict) -> None
        # 用于长时间运行的节点（如训练）报告进度
        self.progress_callback: Optional[callable] = None
        
        # 状态消息回调函数: (message: str) -> None
        # 用于在状态栏显示关键信息
        self.status_callback: Optional[callable] = None
        
        # 初始化端口和参数
        self._setup_ports()
        self._setup_parameters()
        self._init_parameter_values()
    
    @abstractmethod
    def _setup_ports(self):
        """
        设置节点的输入输出端口
        
        子类必须实现此方法，使用 add_input 和 add_output 添加端口。
        """
        pass
    
    def _setup_parameters(self):
        """
        设置节点参数
        
        子类可以覆盖此方法添加参数。
        """
        pass
    
    def _init_parameter_values(self):
        """初始化参数值为默认值"""
        for name, param in self.parameters.items():
            self.parameter_values[name] = param.default_value
    
    def add_input(
        self,
        name: str,
        data_type: DataType,
        display_name: str = None,
        required: bool = True,
        default_value: Any = None,
        description: str = ""
    ):
        """添加输入端口"""
        port = create_input_port(
            name=name,
            data_type=data_type,
            display_name=display_name,
            required=required,
            default_value=default_value,
            description=description
        )
        self.inputs[name] = port
    
    def add_output(
        self,
        name: str,
        data_type: DataType,
        display_name: str = None,
        description: str = ""
    ):
        """添加输出端口"""
        port = create_output_port(
            name=name,
            data_type=data_type,
            display_name=display_name,
            description=description
        )
        self.outputs[name] = port
    
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
        file_filter: str = "",
        default_directory: str = ""
    ):
        """添加节点参数"""
        param = NodeParameter(
            name=name,
            display_name=display_name or name,
            param_type=param_type,
            default_value=default_value,
            description=description,
            min_value=min_value,
            max_value=max_value,
            choices=choices,
            file_filter=file_filter,
            default_directory=default_directory
        )
        self.parameters[name] = param
        self.parameter_values[name] = default_value
    
    def get_parameter(self, name: str) -> Any:
        """获取参数值"""
        return self.parameter_values.get(name)
    
    def set_parameter(self, name: str, value: Any) -> Tuple[bool, str]:
        """
        设置参数值
        
        Returns:
            (success, error_message)
        """
        if name not in self.parameters:
            return False, f"未知参数: {name}"
        
        param = self.parameters[name]
        valid, msg = param.validate(value)
        if not valid:
            return False, msg
        
        self.parameter_values[name] = value
        return True, ""
    
    def get_input_data(self, port_name: str) -> Any:
        """获取输入端口的数据"""
        if port_name not in self.inputs:
            return None
        port = self.inputs[port_name]
        return port.data if port.data is not None else port.default_value
    
    def set_output_data(self, port_name: str, data: Any):
        """设置输出端口的数据"""
        if port_name in self.outputs:
            self.outputs[port_name].data = data
    
    def validate(self) -> Tuple[bool, str]:
        """
        验证节点配置
        
        Returns:
            (is_valid, error_message)
        """
        # 检查必需的输入端口
        for name, port in self.inputs.items():
            if port.required and not port.is_connected and port.default_value is None:
                return False, f"输入端口 '{port.display_name}' 未连接"
        
        # 验证参数
        for name, value in self.parameter_values.items():
            if name in self.parameters:
                valid, msg = self.parameters[name].validate(value)
                if not valid:
                    return False, msg
        
        return True, ""
    
    @abstractmethod
    def execute(self) -> bool:
        """
        执行节点逻辑
        
        子类必须实现此方法。
        应该从输入端口读取数据，处理后写入输出端口。
        
        Returns:
            True 执行成功, False 执行失败
        """
        pass
    
    def reset(self):
        """重置节点状态"""
        self.state = NodeState.IDLE
        self.error_message = ""
        # 清除端口数据
        for port in self.inputs.values():
            port.clear()
        for port in self.outputs.values():
            port.clear()
    
    def report_progress(self, progress: float, message: str = "", data: Dict = None):
        """
        报告节点执行进度
        
        Args:
            progress: 进度值 (0.0 ~ 1.0)
            message: 进度消息
            data: 附加数据（如训练metrics）
        """
        if self.progress_callback:
            self.progress_callback(progress, message, data or {})
    
    def report_status(self, message: str):
        """
        报告状态消息（显示在状态栏）
        
        Args:
            message: 状态消息
        """
        if self.status_callback:
            self.status_callback(message)
    
    def to_dict(self) -> Dict:
        """序列化节点为字典"""
        return {
            "id": self.node_id,
            "type": self.node_type,
            "position": list(self.position),
            "parameters": dict(self.parameter_values),
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'BaseNode':
        """从字典反序列化节点"""
        node = cls(node_id=data.get("id"))
        node.position = tuple(data.get("position", [0, 0]))
        for name, value in data.get("parameters", {}).items():
            node.set_parameter(name, value)
        return node
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(id={self.node_id}, type={self.node_type})>"


# 节点注册表
_node_registry: Dict[str, Type[BaseNode]] = {}


def register_node(node_class: Type[BaseNode]) -> Type[BaseNode]:
    """
    节点注册装饰器
    
    使用方法:
        @register_node
        class MyNode(BaseNode):
            node_type = "my_node"
            ...
    """
    _node_registry[node_class.node_type] = node_class
    logger.debug(f"注册节点: {node_class.node_type} -> {node_class.__name__}")
    return node_class


def get_node_class(node_type: str) -> Optional[Type[BaseNode]]:
    """根据类型获取节点类"""
    return _node_registry.get(node_type)


def get_all_node_types() -> List[str]:
    """获取所有已注册的节点类型"""
    return list(_node_registry.keys())


def get_nodes_by_category(category: NodeCategory) -> List[Type[BaseNode]]:
    """获取指定分类的所有节点类"""
    return [
        cls for cls in _node_registry.values()
        if cls.category == category
    ]


def create_node(node_type: str, node_id: str = None) -> Optional[BaseNode]:
    """根据类型创建节点实例"""
    node_class = get_node_class(node_type)
    if node_class:
        return node_class(node_id=node_id)
    return None

