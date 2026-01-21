# -*- coding: utf-8 -*-
"""
View Module

Provides multi-view switching functionality.
"""

from .workflow_view import WorkflowView
from .preview_view import PreviewView
from .training_view import TrainingView

__all__ = [
    'WorkflowView',
    'PreviewView',
    'TrainingView',
]

