# -*- coding: utf-8 -*-
"""
控制器层

负责协调视图和业务逻辑之间的交互。
"""

from .workflow_controller import WorkflowController
from .training_controller import TrainingController
from .navigation_controller import NavigationController

__all__ = [
    'WorkflowController',
    'TrainingController',
    'NavigationController',
]


