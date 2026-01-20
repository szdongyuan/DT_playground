# -*- coding: utf-8 -*-
"""
预览组件模块

提供各种数据类型的可视化预览组件。
"""

from .base_preview import BasePreviewWidget
from .audio_preview import AudioPreviewWidget
from .feature_1d_preview import Feature1DPreviewWidget
from .feature_2d_preview import Feature2DPreviewWidget
from .label_preview import LabelPreviewWidget
from .metrics_preview import MetricsPreviewWidget

__all__ = [
    'BasePreviewWidget',
    'AudioPreviewWidget',
    'Feature1DPreviewWidget',
    'Feature2DPreviewWidget',
    'LabelPreviewWidget',
    'MetricsPreviewWidget',
]

