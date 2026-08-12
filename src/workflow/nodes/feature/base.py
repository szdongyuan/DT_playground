# -*- coding: utf-8 -*-
"""
Feature Extraction Base Module

Shared data structures and utilities for feature extraction nodes.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

import numpy as np

# Re-export validate_audio_input for convenience
from ..preprocessing import validate_audio_input
from ..data_source import AudioData
from src.ui.i18n import tr_
@dataclass
class FeatureData:
    """
    特征数据包装
    
    数据格式：
    - 多通道特征: (channels, features, frames)
    - 单通道特征: (1, features, frames)
    - 统计特征: (channels, features) 或 (1, features)
    """
    data: np.ndarray        # 特征数据
    feature_type: str       # 特征类型
    sample_rate: int        # 原始采样率
    hop_length: int         # 帧移
    source_file: str = ""   # 源文件路径
    metadata: Dict[str, Any] = field(default_factory=dict)  # Optional feature provenance
    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def channels(self) -> int:
        """返回通道数"""
        return self.data.shape[0]
    
    @property
    def is_mono(self) -> bool:
        """是否为单通道"""
        return self.channels == 1
    
    @property
    def duration(self) -> float:
        """估算时长（秒）"""
        if len(self.data.shape) >= 3:
            # (channels, features, frames)
            n_frames = self.data.shape[-1]
            return n_frames * self.hop_length / self.sample_rate
        elif len(self.data.shape) == 2:
            # 可能是旧格式 (features, frames) 或统计特征 (channels, features)
            n_frames = self.data.shape[-1]
            return n_frames * self.hop_length / self.sample_rate
        return 0.0
    
    def get_channel(self, channel_idx: int) -> np.ndarray:
        """获取指定通道的特征"""
        if channel_idx >= self.channels:
            raise ValueError(
                tr_("Channel index {idx} out of range (total {channels} channels)").format(
                    idx=channel_idx,
                    channels=self.channels,
                )
            )
        return self.data[channel_idx]


def extract_from_audio_or_list(audio_input, extract_func) -> Union[FeatureData, List[FeatureData]]:
    """
    从单个音频或音频列表提取特征
    
    Args:
        audio_input: 单个 AudioData 或 AudioData 列表
        extract_func: 特征提取函数，接受 AudioData 返回 FeatureData
        
    Returns:
        单个 FeatureData 或 FeatureData 列表
    """
    if isinstance(audio_input, list):
        return [extract_func(audio) for audio in audio_input]
    else:
        return extract_func(audio_input)
