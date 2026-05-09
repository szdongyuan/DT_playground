# -*- coding: utf-8 -*-
"""
View Module

Provides multi-view switching functionality.
"""

from .workflow_editor_widget import WorkflowEditorWidget
from .toolbars import ModelToolbar, WorkflowToolbar
from .workflow_view import WorkflowView
from .workflow_tabs_view import WorkflowTabsView
from .preview_view import PreviewView
from .training_view import TrainingView

__all__ = [
    'WorkflowEditorWidget',
    'ModelToolbar',
    'WorkflowToolbar',
    'WorkflowView',
    'WorkflowTabsView',
    'PreviewView',
    'TrainingView',
]

