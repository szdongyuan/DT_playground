# -*- coding: utf-8 -*-
"""
视图模块

提供多视图切换功能。
"""

from .workflow_view import WorkflowView
from .preview_view import PreviewView
from .training_view import TrainingView

__all__ = [
    'WorkflowView',
    'PreviewView',
    'TrainingView',
]

