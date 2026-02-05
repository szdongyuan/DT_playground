# -*- coding: utf-8 -*-
"""
Audio Preview Component

Integrates waveform display, spectrogram, and playback controls.
"""

import logging
from typing import Any, List, Optional

import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QGroupBox, QSplitter, QVBoxLayout, QWidget
)

from src.workflow.port import DataType
from src.ui.i18n import tr_
from src.ui.styles import Styles
from src.ui.widgets.audio_player import AudioPlayerWidget
from src.ui.widgets.spectrogram_widget import SpectrogramWidget
from src.ui.widgets.waveform_widget import WaveformWidget

from .base_preview import BasePreviewWidget, register_preview


logger = logging.getLogger(__name__)


@register_preview
class AudioPreviewWidget(BasePreviewWidget):
    """
    音频预览组件
    
    显示音频的波形、频谱图，并提供播放控制。
    """
    
    supported_types = [DataType.AUDIO]
    display_name = tr_("Audio preview")
    icon = "🎵"
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # 主分割器
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setHandleWidth(2)
        layout.addWidget(splitter)
        
        # 波形显示
        waveform_group = QGroupBox(tr_("Waveform"))
        waveform_group.setStyleSheet(Styles.group_box(Styles.COLORS['blue']))
        waveform_layout = QVBoxLayout(waveform_group)
        waveform_layout.setContentsMargins(4, 4, 4, 4)
        
        self._waveform_widget = WaveformWidget()
        waveform_layout.addWidget(self._waveform_widget)
        splitter.addWidget(waveform_group)
        
        # 频谱图显示
        spectrogram_group = QGroupBox(tr_("Spectrogram"))
        spectrogram_group.setStyleSheet(Styles.group_box(Styles.COLORS['purple']))
        spectrogram_layout = QVBoxLayout(spectrogram_group)
        spectrogram_layout.setContentsMargins(4, 4, 4, 4)
        
        self._spectrogram_widget = SpectrogramWidget()
        spectrogram_layout.addWidget(self._spectrogram_widget)
        splitter.addWidget(spectrogram_group)
        
        # 播放控制
        player_group = QGroupBox(tr_("Playback controls"))
        player_group.setStyleSheet(Styles.group_box(Styles.COLORS['green']))
        player_layout = QVBoxLayout(player_group)
        player_layout.setContentsMargins(4, 4, 4, 4)
        
        self._audio_player = AudioPlayerWidget()
        player_layout.addWidget(self._audio_player)
        layout.addWidget(player_group)
        
        # 设置分割器比例
        splitter.setSizes([300, 300])
    
    def set_data(self, data: Any) -> bool:
        """
        设置音频数据
        
        Args:
            data: AudioData 对象
            
        Returns:
            是否成功设置
        """
        self._current_data = data
        
        if data is None:
            self.clear()
            return False
        
        # 验证数据类型
        if not hasattr(data, 'data') or not hasattr(data, 'sample_rate'):
            logger.warning(f"无效的音频数据类型: {type(data)}")
            return False
        
        try:
            # 更新数据信息
            shape = data.data.shape
            channels = data.channels
            duration = data.duration
            self._update_data_info(
                f"🎵 AUDIO  shape={shape}  ch={channels}  "
                f"sr={data.sample_rate}Hz  {duration:.2f}s"
            )
            
            # 对于多通道音频，显示时混合为单声道以便可视化
            # data.data 形状为 (channels, samples)
            if data.is_mono:
                display_data = data.data[0]  # 取第一通道
            else:
                display_data = np.mean(data.data, axis=0)  # 混合为单声道
            
            # 更新波形
            self._waveform_widget.set_audio(display_data, data.sample_rate)
            
            # 更新频谱图
            self._spectrogram_widget.set_audio(display_data, data.sample_rate)
            
            # 更新播放器
            self._audio_player.set_audio(display_data, data.sample_rate)
            
            self.data_changed.emit()
            return True
            
        except Exception as e:
            logger.error(f"设置音频数据失败: {e}")
            return False
    
    def clear(self):
        """清除显示"""
        self._current_data = None
        self._data_info = ""
        self._waveform_widget.clear()
        self._spectrogram_widget.clear()
        self._audio_player.stop()
    
    @classmethod
    def can_display(cls, data: Any) -> bool:
        """检查是否可以显示该数据"""
        # AudioData 有 data 和 sample_rate 属性，但没有 feature_type
        if hasattr(data, 'data') and hasattr(data, 'sample_rate'):
            # 排除 FeatureData（它也有 sample_rate）
            if hasattr(data, 'feature_type'):
                return False
            return True
        return False

