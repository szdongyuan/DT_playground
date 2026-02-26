# -*- coding: utf-8 -*-
"""
Workflow Data Model

Manages workflow nodes and connections, provides serialization/deserialization functionality.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from src.ui.i18n import tr_
from .connection import Connection
from .node_base import BaseNode, create_node, get_node_class
from .port import DataType


logger = logging.getLogger(__name__)


@dataclass
class WorkflowMetadata:
    """Workflow metadata"""
    name: str = tr_("Untitled workflow")
    description: str = ""
    author: str = ""
    created_at: str = ""
    modified_at: str = ""
    version: str = "1.0"


class Workflow:
    """
    Workflow class
    
    Manages all nodes and connection relationships in a workflow.
    """
    
    def __init__(self, name: str = tr_("Untitled workflow")):
        self.metadata = WorkflowMetadata(name=name)
        self.nodes: Dict[str, BaseNode] = {}
        self.connections: List[Connection] = []
        
        # Update creation time
        self.metadata.created_at = datetime.now().isoformat()
        self.metadata.modified_at = self.metadata.created_at
        
        # Dirty flag
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
        """Mark workflow as modified"""
        self._dirty = True
        self.metadata.modified_at = datetime.now().isoformat()
    
    def mark_saved(self):
        """Mark as saved"""
        self._dirty = False
    
    # ===== Node Management =====
    
    def add_node(self, node: BaseNode) -> bool:
        """
        Add node
        
        Returns:
            Whether addition was successful
        """
        if node.node_id in self.nodes:
            logger.warning(f"Node {node.node_id} already exists")
            return False
        
        self.nodes[node.node_id] = node
        self._mark_dirty()
        logger.debug(f"Added node: {node}")
        return True
    
    def remove_node(self, node_id: str) -> bool:
        """
        Remove node and all its connections
        
        Returns:
            Whether removal was successful
        """
        if node_id not in self.nodes:
            return False
        
        # Remove related connections
        self.connections = [
            conn for conn in self.connections
            if conn.source_node_id != node_id and conn.target_node_id != node_id
        ]
        
        del self.nodes[node_id]
        self._mark_dirty()
        logger.debug(f"Removed node: {node_id}")
        return True
    
    def get_node(self, node_id: str) -> Optional[BaseNode]:
        """Get node"""
        return self.nodes.get(node_id)
    
    def create_and_add_node(self, node_type: str, position: Tuple[float, float] = (0, 0)) -> Optional[BaseNode]:
        """
        Create and add node
        
        Args:
            node_type: Node type
            position: Node position
        
        Returns:
            Created node, None on failure
        """
        node = create_node(node_type)
        if node:
            node.position = position
            if self.add_node(node):
                return node
        return None
    
    # ===== Connection Management =====
    
    def add_connection(self, connection: Connection) -> Tuple[bool, str]:
        """
        Add connection
        
        Returns:
            (success, error_message)
        """
        # Validate nodes exist
        source_node = self.nodes.get(connection.source_node_id)
        target_node = self.nodes.get(connection.target_node_id)
        
        if not source_node:
            return False, tr_("Source node {id} does not exist").format(
                id=connection.source_node_id
            )
        if not target_node:
            return False, tr_("Target node {id} does not exist").format(
                id=connection.target_node_id
            )
        
        # Validate ports exist
        if connection.source_port not in source_node.outputs:
            return False, tr_("Source port {name} does not exist").format(
                name=connection.source_port
            )
        if connection.target_port not in target_node.inputs:
            return False, tr_("Target port {name} does not exist").format(
                name=connection.target_port
            )
        
        # Validate port type compatibility
        source_port = source_node.outputs[connection.source_port]
        target_port = target_node.inputs[connection.target_port]
        
        if not source_port.can_connect_to(target_port):
            return False, tr_("Incompatible port types: {src} -> {dst}").format(
                src=source_port.data_type.value,
                dst=target_port.data_type.value,
            )
        
        # Check if already exists
        if connection in self.connections:
            return False, tr_("Connection already exists")
        
        # Check if target port is already connected (unless multi-connection allowed)
        if not target_port.multi_connection:
            for conn in self.connections:
                if conn.target_node_id == connection.target_node_id and conn.target_port == connection.target_port:
                    return False, tr_("Port {name} is already connected").format(
                        name=target_port.display_name
                    )
        
        # Add connection
        self.connections.append(connection)
        source_port.set_connected(True)
        target_port.set_connected(True)
        self._mark_dirty()
        
        logger.debug(f"Added connection: {connection}")
        return True, ""
    
    def connect(
        self,
        source_node_id: str,
        source_port: str,
        target_node_id: str,
        target_port: str
    ) -> Tuple[bool, str]:
        """Convenience method to create connection"""
        connection = Connection(
            source_node_id=source_node_id,
            source_port=source_port,
            target_node_id=target_node_id,
            target_port=target_port
        )
        return self.add_connection(connection)
    
    def remove_connection(self, connection: Connection) -> bool:
        """Remove connection"""
        if connection not in self.connections:
            return False
        
        self.connections.remove(connection)
        
        # Update port connection status
        self._update_port_connection_status()
        
        self._mark_dirty()
        logger.debug(f"Removed connection: {connection}")
        return True
    
    def _update_port_connection_status(self):
        """Update connection status for all ports"""
        # Reset all port status
        for node in self.nodes.values():
            for port in node.inputs.values():
                port.set_connected(False)
            for port in node.outputs.values():
                port.set_connected(False)
        
        # Update status based on connections
        for conn in self.connections:
            source_node = self.nodes.get(conn.source_node_id)
            target_node = self.nodes.get(conn.target_node_id)
            if source_node and conn.source_port in source_node.outputs:
                source_node.outputs[conn.source_port].set_connected(True)
            if target_node and conn.target_port in target_node.inputs:
                target_node.inputs[conn.target_port].set_connected(True)
    
    def get_incoming_connections(self, node_id: str) -> List[Connection]:
        """Get all incoming connections for a node"""
        return [c for c in self.connections if c.target_node_id == node_id]
    
    def get_outgoing_connections(self, node_id: str) -> List[Connection]:
        """Get all outgoing connections for a node"""
        return [c for c in self.connections if c.source_node_id == node_id]
    
    def disconnect_node(self, node_id: str) -> bool:
        """
        Remove all connections for specified node
        
        Args:
            node_id: Node ID
        
        Returns:
            Whether any connections were removed
        """
        if node_id not in self.nodes:
            return False
        
        # Collect all connections related to this node
        connections_to_remove = [
            conn for conn in self.connections
            if conn.source_node_id == node_id or conn.target_node_id == node_id
        ]
        
        if not connections_to_remove:
            return False
        
        # Remove connections
        for conn in connections_to_remove:
            self.connections.remove(conn)
        
        # Update port connection status
        self._update_port_connection_status()
        
        self._mark_dirty()
        logger.debug(f"Removed all connections for node {node_id}: {len(connections_to_remove)} connections")
        return True
    
    # ===== Validation =====
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate workflow
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        if not self.nodes:
            errors.append(tr_("Workflow is empty"))
            return False, errors
        
        # Validate each node
        for node in self.nodes.values():
            valid, msg = node.validate()
            if not valid:
                errors.append(
                    tr_("Node '{name}' ({id}): {message}").format(
                        name=node.display_name,
                        id=node.node_id,
                        message=msg,
                    )
                )
        
        # Check for disconnected required input ports
        for node in self.nodes.values():
            for port_name, port in node.inputs.items():
                if port.required and not port.is_connected and port.default_value is None:
                    errors.append(
                        tr_("Node '{node}' input port '{port}' is not connected").format(
                            node=node.display_name,
                            port=port.display_name,
                        )
                    )
        
        return len(errors) == 0, errors
    
    # ===== Topology Analysis =====
    
    def get_execution_order(self) -> Tuple[List[str], bool]:
        """
        Get node execution order (topological sort)
        
        Returns:
            (ordered_node_ids, has_cycle)
        """
        # Calculate in-degree for each node
        in_degree: Dict[str, int] = {node_id: 0 for node_id in self.nodes}
        for conn in self.connections:
            if conn.target_node_id in in_degree:
                in_degree[conn.target_node_id] += 1
        
        # Kahn's algorithm
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
        """Get all upstream nodes for specified node"""
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
        """Get all downstream nodes for specified node"""
        downstream = set()
        to_visit = [node_id]
        
        while to_visit:
            current = to_visit.pop()
            for conn in self.get_outgoing_connections(current):
                if conn.target_node_id not in downstream:
                    downstream.add(conn.target_node_id)
                    to_visit.append(conn.target_node_id)
        
        return downstream
    
    # ===== Serialization =====
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary"""
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
        """Deserialize from dictionary"""
        workflow = cls()
        
        # Load metadata
        metadata = data.get("metadata", {})
        workflow.metadata.name = metadata.get("name", tr_("Untitled workflow"))
        workflow.metadata.description = metadata.get("description", "")
        workflow.metadata.author = metadata.get("author", "")
        workflow.metadata.created_at = metadata.get("created_at", "")
        workflow.metadata.modified_at = metadata.get("modified_at", "")
        workflow.metadata.version = data.get("version", "1.0")
        
        # Load nodes
        for node_data in data.get("nodes", []):
            node_type = node_data.get("type")
            node_class = get_node_class(node_type)
            if node_class:
                node = node_class.from_dict(node_data)
                workflow.nodes[node.node_id] = node
            else:
                logger.warning(f"Unknown node type: {node_type}")
        
        # Load connections
        for conn_data in data.get("connections", []):
            conn = Connection.from_dict(conn_data)
            workflow.connections.append(conn)
        
        # Update port status
        workflow._update_port_connection_status()
        
        workflow._dirty = False
        return workflow
    
    def save(self, filepath: str) -> bool:
        """
        Save workflow to file
        
        Args:
            filepath: File path
        
        Returns:
            Whether save was successful
        """
        try:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
            
            self.mark_saved()
            logger.info(f"Workflow saved: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save workflow: {e}")
            return False
    
    @classmethod
    def load(cls, filepath: str) -> Optional['Workflow']:
        """
        Load workflow from file
        
        Args:
            filepath: File path
        
        Returns:
            Workflow object, None on failure
        """
        try:
            path = Path(filepath)
            if not path.exists():
                logger.error(f"File does not exist: {filepath}")
                return None
            
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            workflow = cls.from_dict(data)
            # Attach source path for UI/session features.
            # Some callsites rely on `_file_path` to persist/restore workflow tabs.
            try:
                workflow._file_path = str(path)  # type: ignore[attr-defined]
            except Exception:
                pass
            logger.info(f"Workflow loaded: {filepath}")
            return workflow
        except Exception as e:
            logger.error(f"Failed to load workflow: {e}")
            return None
    
    def __repr__(self) -> str:
        return f"Workflow(name='{self.name}', nodes={len(self.nodes)}, connections={len(self.connections)})"

