# -*- coding: utf-8 -*-
"""
核心抽象层

提供通用的图结构基类、参数定义和事件总线。
"""

from .parameter import Parameter, ParamType
from .graph_base import GraphNodeBase, GraphBase, Connection
from .event_bus import EventBus, get_event_bus

__all__ = [
    'Parameter',
    'ParamType',
    'GraphNodeBase',
    'GraphBase',
    'Connection',
    'EventBus',
    'get_event_bus',
]


