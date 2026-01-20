# -*- coding: utf-8 -*-
"""
工作流数据模型

管理工作流的节点和连接，提供序列化/反序列化功能。
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .connection import Connection
from .node_base import BaseNode, create_node, get_node_class
from .port import DataType


logger = logging.getLogger(__name__)


@dataclass
class WorkflowMetadata:
    """工作流元数据"""
    name: str = "未命名工作流"
    description: str = ""
    author: str = ""
    created_at: str = ""
    modified_at: str = ""
    version: str = "1.0"


class Workflow:
    """
    工作流类
    
    管理工作流中的所有节点和连接关系。
    """
    
    def __init__(self, name: str = "未命名工作流"):
        self.metadata = WorkflowMetadata(name=name)
        self.nodes: Dict[str, BaseNode] = {}
        self.connections: List[Connection] = []
        
        # 更新创建时间
        self.metadata.created_at = datetime.now().isoformat()
        self.metadata.modified_at = self.metadata.created_at
        
        # 脏标记
        self._dirty = False
    
    @property
    def name(self) -> str:
        return self.metadata.name
    
    @name.setter
    def name(self, value: str):
        self.metadata.name = value
        self._mark_dirty()
    
    @property
    def is_dirty(self) -> bool:
        return self._dirty
    
    def _mark_dirty(self):
        """标记工作流已修改"""
        self._dirty = True
        self.metadata.modified_at = datetime.now().isoformat()
    
    def mark_saved(self):
        """标记已保存"""
        self._dirty = False
    
    # ===== 节点管理 =====
    
    def add_node(self, node: BaseNode) -> bool:
        """
        添加节点
        
        Returns:
            是否添加成功
        """
        if node.node_id in self.nodes:
            logger.warning(f"节点 {node.node_id} 已存在")
            return False
        
        self.nodes[node.node_id] = node
        self._mark_dirty()
        logger.debug(f"添加节点: {node}")
        return True
    
    def remove_node(self, node_id: str) -> bool:
        """
        移除节点及其所有连接
        
        Returns:
            是否移除成功
        """
        if node_id not in self.nodes:
            return False
        
        # 移除相关连接
        self.connections = [
            conn for conn in self.connections
            if conn.source_node_id != node_id and conn.target_node_id != node_id
        ]
        
        del self.nodes[node_id]
        self._mark_dirty()
        logger.debug(f"移除节点: {node_id}")
        return True
    
    def get_node(self, node_id: str) -> Optional[BaseNode]:
        """获取节点"""
        return self.nodes.get(node_id)
    
    def create_and_add_node(self, node_type: str, position: Tuple[float, float] = (0, 0)) -> Optional[BaseNode]:
        """
        创建并添加节点
        
        Args:
            node_type: 节点类型
            position: 节点位置
        
        Returns:
            创建的节点，失败返回 None
        """
        node = create_node(node_type)
        if node:
            node.position = position
            if self.add_node(node):
                return node
        return None
    
    # ===== 连接管理 =====
    
    def add_connection(self, connection: Connection) -> Tuple[bool, str]:
        """
        添加连接
        
        Returns:
            (success, error_message)
        """
        # 验证节点存在
        source_node = self.nodes.get(connection.source_node_id)
        target_node = self.nodes.get(connection.target_node_id)
        
        if not source_node:
            return False, f"源节点 {connection.source_node_id} 不存在"
        if not target_node:
            return False, f"目标节点 {connection.target_node_id} 不存在"
        
        # 验证端口存在
        if connection.source_port not in source_node.outputs:
            return False, f"源端口 {connection.source_port} 不存在"
        if connection.target_port not in target_node.inputs:
            return False, f"目标端口 {connection.target_port} 不存在"
        
        # 验证端口类型兼容
        source_port = source_node.outputs[connection.source_port]
        target_port = target_node.inputs[connection.target_port]
        
        if not source_port.can_connect_to(target_port):
            return False, f"端口类型不兼容: {source_port.data_type.value} -> {target_port.data_type.value}"
        
        # 检查是否已存在
        if connection in self.connections:
            return False, "连接已存在"
        
        # 检查目标端口是否已连接（除非允许多连接）
        if not target_port.multi_connection:
            for conn in self.connections:
                if conn.target_node_id == connection.target_node_id and conn.target_port == connection.target_port:
                    return False, f"端口 {target_port.display_name} 已连接"
        
        # 添加连接
        self.connections.append(connection)
        source_port.set_connected(True)
        target_port.set_connected(True)
        self._mark_dirty()
        
        logger.debug(f"添加连接: {connection}")
        return True, ""
    
    def connect(
        self,
        source_node_id: str,
        source_port: str,
        target_node_id: str,
        target_port: str
    ) -> Tuple[bool, str]:
        """创建连接的便捷方法"""
        connection = Connection(
            source_node_id=source_node_id,
            source_port=source_port,
            target_node_id=target_node_id,
            target_port=target_port
        )
        return self.add_connection(connection)
    
    def remove_connection(self, connection: Connection) -> bool:
        """移除连接"""
        if connection not in self.connections:
            return False
        
        self.connections.remove(connection)
        
        # 更新端口连接状态
        self._update_port_connection_status()
        
        self._mark_dirty()
        logger.debug(f"移除连接: {connection}")
        return True
    
    def _update_port_connection_status(self):
        """更新所有端口的连接状态"""
        # 重置所有端口状态
        for node in self.nodes.values():
            for port in node.inputs.values():
                port.set_connected(False)
            for port in node.outputs.values():
                port.set_connected(False)
        
        # 根据连接更新状态
        for conn in self.connections:
            source_node = self.nodes.get(conn.source_node_id)
            target_node = self.nodes.get(conn.target_node_id)
            if source_node and conn.source_port in source_node.outputs:
                source_node.outputs[conn.source_port].set_connected(True)
            if target_node and conn.target_port in target_node.inputs:
                target_node.inputs[conn.target_port].set_connected(True)
    
    def get_incoming_connections(self, node_id: str) -> List[Connection]:
        """获取节点的所有输入连接"""
        return [c for c in self.connections if c.target_node_id == node_id]
    
    def get_outgoing_connections(self, node_id: str) -> List[Connection]:
        """获取节点的所有输出连接"""
        return [c for c in self.connections if c.source_node_id == node_id]
    
    def disconnect_node(self, node_id: str) -> bool:
        """
        移除指定节点的所有连接
        
        Args:
            node_id: 节点ID
        
        Returns:
            是否有连接被移除
        """
        if node_id not in self.nodes:
            return False
        
        # 收集该节点相关的所有连接
        connections_to_remove = [
            conn for conn in self.connections
            if conn.source_node_id == node_id or conn.target_node_id == node_id
        ]
        
        if not connections_to_remove:
            return False
        
        # 移除连接
        for conn in connections_to_remove:
            self.connections.remove(conn)
        
        # 更新端口连接状态
        self._update_port_connection_status()
        
        self._mark_dirty()
        logger.debug(f"移除节点 {node_id} 的所有连接: {len(connections_to_remove)} 个")
        return True
    
    # ===== 验证 =====
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        验证工作流
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        if not self.nodes:
            errors.append("工作流为空")
            return False, errors
        
        # 验证每个节点
        for node in self.nodes.values():
            valid, msg = node.validate()
            if not valid:
                errors.append(f"节点 '{node.display_name}' ({node.node_id}): {msg}")
        
        # 检查是否有孤立的必需输入端口
        for node in self.nodes.values():
            for port_name, port in node.inputs.items():
                if port.required and not port.is_connected and port.default_value is None:
                    errors.append(
                        f"节点 '{node.display_name}' 的输入端口 '{port.display_name}' 未连接"
                    )
        
        return len(errors) == 0, errors
    
    # ===== 拓扑分析 =====
    
    def get_execution_order(self) -> Tuple[List[str], bool]:
        """
        获取节点执行顺序（拓扑排序）
        
        Returns:
            (ordered_node_ids, has_cycle)
        """
        # 计算每个节点的入度
        in_degree: Dict[str, int] = {node_id: 0 for node_id in self.nodes}
        for conn in self.connections:
            if conn.target_node_id in in_degree:
                in_degree[conn.target_node_id] += 1
        
        # Kahn's算法
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            node_id = queue.pop(0)
            result.append(node_id)
            
            for conn in self.get_outgoing_connections(node_id):
                in_degree[conn.target_node_id] -= 1
                if in_degree[conn.target_node_id] == 0:
                    queue.append(conn.target_node_id)
        
        has_cycle = len(result) != len(self.nodes)
        return result, has_cycle
    
    def get_upstream_nodes(self, node_id: str) -> Set[str]:
        """获取指定节点的所有上游节点"""
        upstream = set()
        to_visit = [node_id]
        
        while to_visit:
            current = to_visit.pop()
            for conn in self.get_incoming_connections(current):
                if conn.source_node_id not in upstream:
                    upstream.add(conn.source_node_id)
                    to_visit.append(conn.source_node_id)
        
        return upstream
    
    def get_downstream_nodes(self, node_id: str) -> Set[str]:
        """获取指定节点的所有下游节点"""
        downstream = set()
        to_visit = [node_id]
        
        while to_visit:
            current = to_visit.pop()
            for conn in self.get_outgoing_connections(current):
                if conn.target_node_id not in downstream:
                    downstream.add(conn.target_node_id)
                    to_visit.append(conn.target_node_id)
        
        return downstream
    
    # ===== 序列化 =====
    
    def to_dict(self) -> Dict:
        """序列化为字典"""
        return {
            "version": self.metadata.version,
            "metadata": {
                "name": self.metadata.name,
                "description": self.metadata.description,
                "author": self.metadata.author,
                "created_at": self.metadata.created_at,
                "modified_at": self.metadata.modified_at,
            },
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "connections": [conn.to_dict() for conn in self.connections],
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Workflow':
        """从字典反序列化"""
        workflow = cls()
        
        # 加载元数据
        metadata = data.get("metadata", {})
        workflow.metadata.name = metadata.get("name", "未命名工作流")
        workflow.metadata.description = metadata.get("description", "")
        workflow.metadata.author = metadata.get("author", "")
        workflow.metadata.created_at = metadata.get("created_at", "")
        workflow.metadata.modified_at = metadata.get("modified_at", "")
        workflow.metadata.version = data.get("version", "1.0")
        
        # 加载节点
        for node_data in data.get("nodes", []):
            node_type = node_data.get("type")
            node_class = get_node_class(node_type)
            if node_class:
                node = node_class.from_dict(node_data)
                workflow.nodes[node.node_id] = node
            else:
                logger.warning(f"未知节点类型: {node_type}")
        
        # 加载连接
        for conn_data in data.get("connections", []):
            conn = Connection.from_dict(conn_data)
            workflow.connections.append(conn)
        
        # 更新端口状态
        workflow._update_port_connection_status()
        
        workflow._dirty = False
        return workflow
    
    def save(self, filepath: str) -> bool:
        """
        保存工作流到文件
        
        Args:
            filepath: 文件路径
        
        Returns:
            是否保存成功
        """
        try:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
            
            self.mark_saved()
            logger.info(f"工作流已保存: {filepath}")
            return True
        except Exception as e:
            logger.error(f"保存工作流失败: {e}")
            return False
    
    @classmethod
    def load(cls, filepath: str) -> Optional['Workflow']:
        """
        从文件加载工作流
        
        Args:
            filepath: 文件路径
        
        Returns:
            工作流对象，失败返回 None
        """
        try:
            path = Path(filepath)
            if not path.exists():
                logger.error(f"文件不存在: {filepath}")
                return None
            
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            workflow = cls.from_dict(data)
            logger.info(f"工作流已加载: {filepath}")
            return workflow
        except Exception as e:
            logger.error(f"加载工作流失败: {e}")
            return None
    
    def __repr__(self) -> str:
        return f"Workflow(name='{self.name}', nodes={len(self.nodes)}, connections={len(self.connections)})"

