"""
Custom UI Widgets Module
"""

from src.ui.widgets.audio_player import AudioPlayerWidget
from src.ui.widgets.spectrogram_widget import SpectrogramWidget
from src.ui.widgets.waveform_widget import WaveformWidget

# 预览组件
from src.ui.widgets.preview import (
    BasePreviewWidget,
    AudioPreviewWidget,
    Feature1DPreviewWidget,
    Feature2DPreviewWidget,
    LabelPreviewWidget,
    MetricsPreviewWidget,
)

__all__ = [
    "AudioPlayerWidget",
    "SpectrogramWidget",
    "WaveformWidget",
    # 预览组件
    "BasePreviewWidget",
    "AudioPreviewWidget",
    "Feature1DPreviewWidget",
    "Feature2DPreviewWidget",
    "LabelPreviewWidget",
    "MetricsPreviewWidget",
]
