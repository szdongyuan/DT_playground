# -*- coding: utf-8 -*-
"""
Spectral Contrast Feature Extraction Node

Extracts spectral contrast features from audio signals.
Spectral contrast measures the difference between peaks and valleys
in the spectrum, useful for distinguishing different timbres.
"""

import numpy as np

from ....node_base import BaseNode, NodeCategory, register_node
from ....port import DataType
from ..base import FeatureData, extract_from_audio_or_list, validate_audio_input, AudioData
from src.ui.i18n import tr_
# Import unified feature extractor
from src.audio.features import FeatureExtractor


@register_node
class SpectralContrastNode(BaseNode):
    """
    频谱对比度节点
    
    提取频谱对比度特征。频谱对比度衡量频谱峰值和谷值之间的差异，
    对于区分不同的音色特别有效。
    """
    
    node_type = "spectral_contrast"
    display_name = tr_("Spectral contrast")
    category = NodeCategory.FEATURE
    subcategory = tr_("2D features")
    description = tr_("Extract spectral contrast")
    icon = "📊"
    
    def _setup_ports(self):
        self.add_input("audio", DataType.AUDIO, tr_("Audio"))
        self.add_output("feature", DataType.FEATURE, tr_("Spectral contrast"))
    
    def _setup_parameters(self):
        self.add_parameter(
            "n_fft", "int", 2048,
            display_name=tr_("FFT window size"),
            min_value=256, max_value=8192
        )
        self.add_parameter(
            "hop_length", "int", 512,
            display_name=tr_("Hop length"),
            min_value=64, max_value=2048
        )
        self.add_parameter(
            "n_bands", "int", 6,
            display_name=tr_("Bands"),
            min_value=2, max_value=12,
            description=tr_("Number of sub-bands (output dimension is n_bands + 1)"),
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
        
        n_fft = self.get_parameter("n_fft")
        hop_length = self.get_parameter("hop_length")
        n_bands = self.get_parameter("n_bands")
        
        def extract_contrast(audio: AudioData) -> FeatureData:
            import librosa
            
            # Extract spectral contrast for each channel
            channel_features = []
            for ch in range(audio.channels):
                ch_data = audio.get_channel(ch)
                contrast = librosa.feature.spectral_contrast(
                    y=ch_data,
                    sr=audio.sample_rate,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    n_bands=n_bands
                )
                channel_features.append(contrast)
            
            # Stack as (channels, n_bands+1, frames)
            stacked = np.stack(channel_features, axis=0)
            
            return FeatureData(
                data=stacked.astype(np.float32),
                feature_type="spectral_contrast",
                sample_rate=audio.sample_rate,
                hop_length=hop_length,
                source_file=audio.file_path
            )
        
        result = extract_from_audio_or_list(audio_input, extract_contrast)
        self.set_output_data("feature", result)
        return True
