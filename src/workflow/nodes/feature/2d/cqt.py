# -*- coding: utf-8 -*-
"""
CQT Feature Extraction Node

Extracts Constant-Q Transform spectrogram from audio signals.
CQT provides logarithmically-spaced frequency resolution, making it
particularly suitable for music signal analysis.
"""

import numpy as np

from ...node_base import BaseNode, NodeCategory, register_node
from ...port import DataType
from ..base import FeatureData, extract_from_audio_or_list, validate_audio_input, AudioData

# Import unified feature extractor
from src.audio.features import FeatureExtractor


@register_node
class CQTNode(BaseNode):
    """
    CQT节点
    
    提取常数Q变换频谱。CQT的频率分辨率随频率呈对数变化，
    特别适合音乐信号分析。
    """
    
    node_type = "cqt"
    display_name = "CQT频谱"
    category = NodeCategory.FEATURE
    description = "提取CQT常数Q变换频谱"
    icon = "🎵"
    
    def _setup_ports(self):
        self.add_input("audio", DataType.AUDIO, "音频")
        self.add_output("feature", DataType.FEATURE, "CQT频谱")
    
    def _setup_parameters(self):
        self.add_parameter(
            "n_bins", "int", 84,
            display_name="频带数",
            min_value=12, max_value=168,
            description="CQT频带数（默认84，即7个八度×12半音）"
        )
        self.add_parameter(
            "bins_per_octave", "int", 12,
            display_name="每八度频带数",
            min_value=6, max_value=48,
            description="每个八度的频带数（默认12，对应半音）"
        )
        self.add_parameter(
            "hop_length", "int", 512,
            display_name="帧移",
            min_value=64, max_value=2048
        )
        self.add_parameter(
            "fmin", "float", 32.7,
            display_name="最低频率(Hz)",
            min_value=10.0, max_value=500.0,
            description="最低频率（默认C1，约32.7Hz）"
        )
        self.add_parameter(
            "to_db", "bool", True,
            display_name="转换为dB",
            description="将幅度谱转换为dB刻度"
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
        
        n_bins = self.get_parameter("n_bins")
        bins_per_octave = self.get_parameter("bins_per_octave")
        hop_length = self.get_parameter("hop_length")
        fmin = self.get_parameter("fmin")
        to_db = self.get_parameter("to_db")
        
        def extract_cqt(audio: AudioData) -> FeatureData:
            # Use unified FeatureExtractor
            extractor = FeatureExtractor(
                sample_rate=audio.sample_rate,
                hop_length=hop_length,
                n_bins=n_bins,
                bins_per_octave=bins_per_octave
            )
            
            # Extract CQT for each channel
            channel_features = []
            for ch in range(audio.channels):
                ch_data = audio.get_channel(ch)
                cqt = extractor.extract_cqt(ch_data, to_db=to_db, fmin=fmin)
                channel_features.append(cqt)
            
            # Stack as (channels, n_bins, frames)
            stacked = np.stack(channel_features, axis=0)
            
            return FeatureData(
                data=stacked.astype(np.float32),
                feature_type="cqt",
                sample_rate=audio.sample_rate,
                hop_length=hop_length,
                source_file=audio.file_path
            )
        
        result = extract_from_audio_or_list(audio_input, extract_cqt)
        self.set_output_data("feature", result)
        return True
