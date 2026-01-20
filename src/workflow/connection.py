# -*- coding: utf-8 -*-
"""
连接定义

定义节点之间的连接关系。
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class Connection:
    """
    节点连接
    
    表示从一个节点的输出端口到另一个节点的输入端口的连接。
    """
    
    source_node_id: str         # 源节点ID
    source_port: str            # 源端口名称（输出端口）
    target_node_id: str         # 目标节点ID
    target_port: str            # 目标端口名称（输入端口）
    
    @property
    def source_key(self) -> str:
        """获取源端口的唯一标识"""
        return f"{self.source_node_id}.{self.source_port}"
    
    @property
    def target_key(self) -> str:
        """获取目标端口的唯一标识"""
        return f"{self.target_node_id}.{self.target_port}"
    
    def to_dict(self) -> Dict:
        """序列化为字典"""
        return {
            "source": {
                "node": self.source_node_id,
                "port": self.source_port
            },
            "target": {
                "node": self.target_node_id,
                "port": self.target_port
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Connection':
        """从字典反序列化"""
        source = data.get("source", {})
        target = data.get("target", {})
        return cls(
            source_node_id=source.get("node", ""),
            source_port=source.get("port", ""),
            target_node_id=target.get("node", ""),
            target_port=target.get("port", "")
        )
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Connection):
            return False
        return (
            self.source_node_id == other.source_node_id and
            self.source_port == other.source_port and
            self.target_node_id == other.target_node_id and
            self.target_port == other.target_port
        )
    
    def __hash__(self) -> int:
        return hash((
            self.source_node_id,
            self.source_port,
            self.target_node_id,
            self.target_port
        ))
    
    def __repr__(self) -> str:
        return f"Connection({self.source_key} -> {self.target_key})"

