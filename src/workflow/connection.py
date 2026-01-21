# -*- coding: utf-8 -*-
"""
Connection Definition

Defines the connection relationships between nodes.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class Connection:
    """
    Node Connection
    
    Represents a connection from one node's output port to another node's input port.
    """
    
    source_node_id: str         # Source node ID
    source_port: str            # Source port name (output port)
    target_node_id: str         # Target node ID
    target_port: str            # Target port name (input port)
    
    @property
    def source_key(self) -> str:
        """Get unique identifier for source port"""
        return f"{self.source_node_id}.{self.source_port}"
    
    @property
    def target_key(self) -> str:
        """Get unique identifier for target port"""
        return f"{self.target_node_id}.{self.target_port}"
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary"""
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
        """Deserialize from dictionary"""
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

