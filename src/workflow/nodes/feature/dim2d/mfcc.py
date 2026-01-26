# -*- coding: utf-8 -*-
"""
MFCC Feature Extraction Node

Extracts Mel-frequency cepstral coefficients from audio signals.
"""

import numpy as np

from ....node_base import BaseNode, NodeCategory, register_node
from ....port import DataType
from ..base import FeatureData, extract_from_audio_or_list, validate_audio_input, AudioData

# Import unified feature extractor
from src.audio.features import FeatureExtractor


@register_node
class MFCCNode(BaseNode):
    """
    MFCC节点
    
    提取梅尔频率倒谱系数。
    """
    
    node_type = "mfcc"
    display_name = "MFCC"
    category = NodeCategory.FEATURE
    subcategory = "二维特征 (2D)"
    description = "提取MFCC特征"
    icon = "📊"
    
    def _setup_ports(self):
        self.add_input("audio", DataType.AUDIO, "音频")
        self.add_output("feature", DataType.FEATURE, "MFCC")
    
    def _setup_parameters(self):
        self.add_parameter(
            "n_mfcc", "int", 13,
            display_name="MFCC系数数",
            min_value=1, max_value=40
        )
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
            "include_delta", "bool", False,
            display_name="包含一阶差分"
        )
        self.add_parameter(
            "include_delta2", "bool", False,
            display_name="包含二阶差分"
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
        
        n_mfcc = self.get_parameter("n_mfcc")
        n_mels = self.get_parameter("n_mels")
        n_fft = self.get_parameter("n_fft")
        hop_length = self.get_parameter("hop_length")
        include_delta = self.get_parameter("include_delta")
        include_delta2 = self.get_parameter("include_delta2")
        
        def extract_mfcc(audio: AudioData) -> FeatureData:
            # Use unified FeatureExtractor
            extractor = FeatureExtractor(
                sample_rate=audio.sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                n_mfcc=n_mfcc
            )
            
            # Extract MFCC for each channel
            channel_features = []
            for ch in range(audio.channels):
                ch_data = audio.get_channel(ch)
                mfcc = extractor.extract_mfcc(
                    ch_data,
                    delta=include_delta,
                    delta_delta=include_delta2
                )
                channel_features.append(mfcc)
            
            # Stack as (channels, n_features, frames)
            stacked = np.stack(channel_features, axis=0)
            
            return FeatureData(
                data=stacked.astype(np.float32),
                feature_type="mfcc",
                sample_rate=audio.sample_rate,
                hop_length=hop_length,
                source_file=audio.file_path
            )
        
        result = extract_from_audio_or_list(audio_input, extract_mfcc)
        self.set_output_data("feature", result)
        return True
