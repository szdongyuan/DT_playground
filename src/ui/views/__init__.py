# -*- coding: utf-8 -*-
"""
View Module

Provides multi-view switching functionality.
"""

from .workflow_editor_widget import WorkflowEditorWidget
from .workflow_toolbar import WorkflowToolbar
from .workflow_view import WorkflowView
from .workflow_tabs_view import WorkflowTabsView
from .preview_view import PreviewView
from .training_view import TrainingView

__all__ = [
    'WorkflowEditorWidget',
    'WorkflowToolbar',
    'WorkflowView',
    'WorkflowTabsView',
    'PreviewView',
    'TrainingView',
]

