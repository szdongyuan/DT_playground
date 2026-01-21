# -*- coding: utf-8 -*-
"""
Workflow Engine Module

Provides core functionality for node-based data processing and training workflows.
"""

from .connection import Connection
from .engine import WorkflowEngine
from .node_base import BaseNode, NodeCategory
from .port import DataType, Port
from .workflow import Workflow

# Import nodes module to trigger node registration
from . import nodes

__all__ = [
    'BaseNode',
    'Connection',
    'DataType',
    'NodeCategory',
    'Port',
    'Workflow',
    'WorkflowEngine',
    'nodes',
]

