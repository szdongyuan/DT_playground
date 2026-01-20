# -*- coding: utf-8 -*-
"""
通用图结构基类

为工作流节点和模型层提供统一的图结构抽象。
"""

import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Type

from .parameter import Parameter, ParamType


logger = logging.getLogger(__name__)


@dataclass
class Connection:
    """
    通用连接定义
    
    表示图中两个节点之间的连接关系。
    """
    source_id: str          # 源节点ID
    target_id: str          # 目标节点ID
    source_port: str = ""   # 源端口名称（可选，用于多端口节点）
    target_port: str = ""   # 目标端口名称（可选）
    
    def to_dict(self) -> dict:
        """序列化为字典"""
        data = {
            "source": self.source_id,
            "target": self.target_id,
        }
        if self.source_port:
            data["source_port"] = self.source_port
        if self.target_port:
            data["target_port"] = self.target_port
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Connection':
        """从字典反序列化"""
        return cls(
            source_id=data["source"],
            target_id=data["target"],
            source_port=data.get("source_port", ""),
            target_port=data.get("target_port", ""),
        )
    
    def __eq__(self, other):
        if not isinstance(other, Connection):
            return False
        return (self.source_id == other.source_id and 
                self.target_id == other.target_id and
                self.source_port == other.source_port and
                self.target_port == other.target_port)
    
    def __hash__(self):
        return hash((self.source_id, self.target_id, self.source_port, self.target_port))


class GraphNodeBase(ABC):
    """
    图节点基类
    
    为 BaseNode（工作流节点）和 LayerNode（模型层）提供通用的基础功能。
    子类需要实现 _setup_parameters 方法。
    """
    
    # 类属性 - 子类需要覆盖
    node_type: str = "base"             # 节点类型标识
    display_name: str = "Base Node"     # 显示名称
    description: str = ""               # 节点描述
    icon: str = "📦"                     # 节点图标
    
    def __init__(self, node_id: str = None):
        """
        初始化节点
        
        Args:
            node_id: 节点唯一标识，默认自动生成
        """
        self.node_id = node_id or str(uuid.uuid4())[:8]
        self.parameters: Dict[str, Parameter] = {}
        self.parameter_values: Dict[str, Any] = {}
        
        # 初始化参数
        self._setup_parameters()
        self._init_parameter_values()
    
    @abstractmethod
    def _setup_parameters(self):
        """
        设置节点参数
        
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
        param_type: ParamType,
        default_value: Any,
        display_name: str = None,
        description: str = "",
        min_value: Any = None,
        max_value: Any = None,
        choices: List[Any] = None,
        file_filter: str = "",
        required: bool = True
    ):
        """添加节点参数"""
        param = Parameter(
            name=name,
            display_name=display_name or name,
            param_type=param_type,
            default_value=default_value,
            description=description,
            min_value=min_value,
            max_value=max_value,
            choices=choices,
            file_filter=file_filter,
            required=required
        )
        self.parameters[name] = param
        self.parameter_values[name] = default_value
    
    def get_parameter(self, name: str) -> Any:
        """获取参数值"""
        return self.parameter_values.get(name)
    
    def set_parameter(self, name: str, value: Any) -> Tuple[bool, str]:
        """
        设置参数值
        
        Args:
            name: 参数名称
            value: 参数值
            
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
    
    def validate(self) -> Tuple[bool, str]:
        """
        验证节点配置
        
        Returns:
            (is_valid, error_message)
        """
        for name, value in self.parameter_values.items():
            if name in self.parameters:
                valid, msg = self.parameters[name].validate(value)
                if not valid:
                    return False, msg
        return True, ""
    
    def to_dict(self) -> dict:
        """序列化节点为字典（基础版本，子类可扩展）"""
        return {
            "id": self.node_id,
            "type": self.node_type,
            "parameters": dict(self.parameter_values),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'GraphNodeBase':
        """从字典反序列化节点"""
        node = cls(node_id=data.get("id"))
        for name, value in data.get("parameters", {}).items():
            node.set_parameter(name, value)
        return node
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(id={self.node_id}, type={self.node_type})>"


class GraphBase(ABC):
    """
    图容器基类
    
    为 Workflow 和 ModelGraph 提供通用的图管理功能。
    """
    
    def __init__(self, name: str = "未命名"):
        self.name = name
        self.nodes: Dict[str, GraphNodeBase] = {}
        self.connections: List[Connection] = []
        self._dirty = False
    
    @property
    def is_dirty(self) -> bool:
        """是否已修改"""
        return self._dirty
    
    def _mark_dirty(self):
        """标记为已修改"""
        self._dirty = True
    
    def mark_saved(self):
        """标记为已保存"""
        self._dirty = False
    
    # ===== 节点管理 =====
    
    def add_node(self, node: GraphNodeBase) -> bool:
        """
        添加节点
        
        Args:
            node: 要添加的节点
            
        Returns:
            是否添加成功
        """
        if node.node_id in self.nodes:
            logger.warning(f"节点 {node.node_id} 已存在")
            return False
        
        self.nodes[node.node_id] = node
        self._mark_dirty()
        return True
    
    def remove_node(self, node_id: str) -> bool:
        """
        移除节点及其所有连接
        
        Args:
            node_id: 节点ID
            
        Returns:
            是否移除成功
        """
        if node_id not in self.nodes:
            return False
        
        # 移除相关连接
        self.connections = [
            conn for conn in self.connections
            if conn.source_id != node_id and conn.target_id != node_id
        ]
        
        del self.nodes[node_id]
        self._mark_dirty()
        return True
    
    def get_node(self, node_id: str) -> Optional[GraphNodeBase]:
        """获取节点"""
        return self.nodes.get(node_id)
    
    # ===== 连接管理 =====
    
    def add_connection(self, connection: Connection) -> Tuple[bool, str]:
        """
        添加连接
        
        Args:
            connection: 连接对象
            
        Returns:
            (success, error_message)
        """
        # 验证节点存在
        if connection.source_id not in self.nodes:
            return False, f"源节点 {connection.source_id} 不存在"
        if connection.target_id not in self.nodes:
            return False, f"目标节点 {connection.target_id} 不存在"
        
        # 检查自连接
        if connection.source_id == connection.target_id:
            return False, "不能自连接"
        
        # 检查是否已存在
        if connection in self.connections:
            return False, "连接已存在"
        
        self.connections.append(connection)
        self._mark_dirty()
        return True, ""
    
    def remove_connection(self, connection: Connection) -> bool:
        """移除连接"""
        if connection not in self.connections:
            return False
        
        self.connections.remove(connection)
        self._mark_dirty()
        return True
    
    def connect(
        self,
        source_id: str,
        target_id: str,
        source_port: str = "",
        target_port: str = ""
    ) -> Tuple[bool, str]:
        """创建连接的便捷方法"""
        connection = Connection(
            source_id=source_id,
            target_id=target_id,
            source_port=source_port,
            target_port=target_port
        )
        return self.add_connection(connection)
    
    def get_incoming_connections(self, node_id: str) -> List[Connection]:
        """获取节点的所有输入连接"""
        return [c for c in self.connections if c.target_id == node_id]
    
    def get_outgoing_connections(self, node_id: str) -> List[Connection]:
        """获取节点的所有输出连接"""
        return [c for c in self.connections if c.source_id == node_id]
    
    # ===== 拓扑分析 =====
    
    def get_execution_order(self) -> Tuple[List[str], bool]:
        """
        获取节点执行顺序（拓扑排序）
        
        使用 Kahn's 算法进行拓扑排序。
        
        Returns:
            (ordered_node_ids, has_cycle)
        """
        # 计算每个节点的入度
        in_degree: Dict[str, int] = {node_id: 0 for node_id in self.nodes}
        for conn in self.connections:
            if conn.target_id in in_degree:
                in_degree[conn.target_id] += 1
        
        # Kahn's算法
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            node_id = queue.pop(0)
            result.append(node_id)
            
            for conn in self.get_outgoing_connections(node_id):
                in_degree[conn.target_id] -= 1
                if in_degree[conn.target_id] == 0:
                    queue.append(conn.target_id)
        
        has_cycle = len(result) != len(self.nodes)
        return result, has_cycle
    
    def get_input_nodes(self) -> List[str]:
        """获取输入节点（没有输入连接的节点）"""
        has_input = set(conn.target_id for conn in self.connections)
        return [nid for nid in self.nodes if nid not in has_input]
    
    def get_output_nodes(self) -> List[str]:
        """获取输出节点（没有输出连接的节点）"""
        has_output = set(conn.source_id for conn in self.connections)
        return [nid for nid in self.nodes if nid not in has_output]
    
    def get_upstream_nodes(self, node_id: str) -> Set[str]:
        """获取指定节点的所有上游节点"""
        upstream = set()
        to_visit = [node_id]
        
        while to_visit:
            current = to_visit.pop()
            for conn in self.get_incoming_connections(current):
                if conn.source_id not in upstream:
                    upstream.add(conn.source_id)
                    to_visit.append(conn.source_id)
        
        return upstream
    
    def get_downstream_nodes(self, node_id: str) -> Set[str]:
        """获取指定节点的所有下游节点"""
        downstream = set()
        to_visit = [node_id]
        
        while to_visit:
            current = to_visit.pop()
            for conn in self.get_outgoing_connections(current):
                if conn.target_id not in downstream:
                    downstream.add(conn.target_id)
                    to_visit.append(conn.target_id)
        
        return downstream
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        验证图结构
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        if not self.nodes:
            errors.append("图为空")
        
        # 验证每个节点
        for node in self.nodes.values():
            valid, msg = node.validate()
            if not valid:
                errors.append(f"节点 '{node.display_name}' ({node.node_id}): {msg}")
        
        # 检查循环
        _, has_cycle = self.get_execution_order()
        if has_cycle:
            errors.append("存在循环依赖")
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> dict:
        """序列化图为字典（基础版本，子类可扩展）"""
        return {
            "name": self.name,
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "connections": [conn.to_dict() for conn in self.connections],
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', nodes={len(self.nodes)}, connections={len(self.connections)})"


# ===== 注册表工具函数 =====

def create_registry():
    """
    创建节点/层注册表
    
    Returns:
        (registry_dict, register_decorator, get_class_func, create_instance_func)
    
    使用示例:
        _registry, register_node, get_node_class, create_node = create_registry()
        
        @register_node
        class MyNode(BaseNode):
            node_type = "my_node"
    """
    registry: Dict[str, Type[GraphNodeBase]] = {}
    
    def register(cls: Type[GraphNodeBase]) -> Type[GraphNodeBase]:
        """注册装饰器"""
        registry[cls.node_type] = cls
        logger.debug(f"注册: {cls.node_type} -> {cls.__name__}")
        return cls
    
    def get_class(node_type: str) -> Optional[Type[GraphNodeBase]]:
        """根据类型获取类"""
        return registry.get(node_type)
    
    def create_instance(node_type: str, node_id: str = None) -> Optional[GraphNodeBase]:
        """根据类型创建实例"""
        cls = get_class(node_type)
        if cls:
            return cls(node_id=node_id)
        return None
    
    def get_all_types() -> List[str]:
        """获取所有已注册的类型"""
        return list(registry.keys())
    
    return registry, register, get_class, create_instance, get_all_types


