# -*- coding: utf-8 -*-
"""
Mel Spectrogram Feature Extraction Node

Extracts Mel spectrogram features from audio signals.
"""

import numpy as np

from ....node_base import BaseNode, NodeCategory, register_node
from ....port import DataType
from ..base import FeatureData, extract_from_audio_or_list, validate_audio_input, AudioData

# Import unified feature extractor
from src.audio.features import FeatureExtractor


@register_node
class MelSpectrogramNode(BaseNode):
    """
    Mel频谱图节点
    
    提取Mel频谱图特征。
    """
    
    node_type = "mel_spectrogram"
    display_name = "Mel频谱图"
    category = NodeCategory.FEATURE
    description = "提取Mel频谱图特征"
    icon = "📈"
    
    def _setup_ports(self):
        self.add_input("audio", DataType.AUDIO, "音频")
        self.add_output("feature", DataType.FEATURE, "Mel频谱")
    
    def _setup_parameters(self):
        self.add_parameter(
            "n_mels", "int", 128,
            display_name="Mel滤波器数",
            min_value=20, max_value=256
        )
        self.add_parameter(
            "n_fft", "int", 2048,
            display_name="FFT窗口大小",
            min_value=256, max_value=8192
        )
        self.add_parameter(
            "hop_length", "int", 512,
            display_name="帧移",
            min_value=64, max_value=2048
        )
        self.add_parameter(
            "fmin", "float", 0.0,
            display_name="最低频率(Hz)",
            min_value=0.0, max_value=1000.0
        )
        self.add_parameter(
            "fmax", "float", 8000.0,
            display_name="最高频率(Hz)",
            min_value=1000.0, max_value=22050.0
        )
        self.add_parameter(
            "power_to_db", "bool", True,
            display_name="转换为dB",
            description="将功率谱转换为dB刻度"
        )
    
    def execute(self) -> bool:
        try:
            audio_input = validate_audio_input(
                self.get_input_data("audio"),
                self.display_name
            )
        except (ValueError, TypeError) as e:
            self.error_message = str(e)
            return False
        
        n_mels = self.get_parameter("n_mels")
        n_fft = self.get_parameter("n_fft")
        hop_length = self.get_parameter("hop_length")
        fmin = self.get_parameter("fmin")
        fmax = self.get_parameter("fmax")
        power_to_db = self.get_parameter("power_to_db")
        
        def extract_mel(audio: AudioData) -> FeatureData:
            # Use unified FeatureExtractor
            extractor = FeatureExtractor(
                sample_rate=audio.sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels
            )
            
            # Extract Mel spectrogram for each channel
            channel_features = []
            for ch in range(audio.channels):
                ch_data = audio.get_channel(ch)
                mel = extractor.extract_mel_spectrogram(ch_data, to_db=power_to_db)
                channel_features.append(mel)
            
            # Stack as (channels, n_mels, frames)
            stacked = np.stack(channel_features, axis=0)
            
            return FeatureData(
                data=stacked.astype(np.float32),
                feature_type="mel_spectrogram",
                sample_rate=audio.sample_rate,
                hop_length=hop_length,
                source_file=audio.file_path
            )
        
        result = extract_from_audio_or_list(audio_input, extract_mel)
        self.set_output_data("feature", result)
        return True
