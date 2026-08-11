# -*- coding: utf-8 -*-
"""
Spectral Flatness Feature Extraction Node

Extracts spectral flatness from audio signals.
Spectral flatness measures how tone-like vs noise-like a signal is.
Values close to 1 indicate white noise (flat spectrum),
values close to 0 indicate tonal signals.
"""

import numpy as np

from ....node_base import BaseNode, NodeCategory, register_node
from ....port import DataType
from ..base import FeatureData, extract_from_audio_or_list, validate_audio_input, AudioData
from src.ui.i18n import tr_
# Import unified feature extractor
from src.audio.features import FeatureExtractor


@register_node
class SpectralFlatnessNode(BaseNode):
    """
    频谱平坦度节点
    
    提取频谱平坦度特征。频谱平坦度衡量信号的"噪声性"与"调性"程度。
    值接近1表示白噪声（平坦频谱），值接近0表示调性信号。
    """
    
    node_type = "spectral_flatness"
    display_name = tr_("Spectral flatness")
    category = NodeCategory.FEATURE
    subcategory = tr_("1D features")
    subcategory_order = 10
    palette_order = 40
    description = tr_("Extract spectral flatness")
    icon = "📏"
    
    def _setup_ports(self):
        self.add_input("audio", DataType.AUDIO, tr_("Audio"))
        self.add_output("feature", DataType.FEATURE, tr_("Spectral flatness"))
    
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
            "aggregate", "choice", "none",
            display_name=tr_("Aggregation"),
            choices=["none", "mean", "std", "mean_std"],
            description=tr_("Aggregate over the time axis"),
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
        aggregate = self.get_parameter("aggregate")
        
        def extract_flatness(audio: AudioData) -> FeatureData:
            # Use unified FeatureExtractor
            extractor = FeatureExtractor(
                sample_rate=audio.sample_rate,
                n_fft=n_fft,
                hop_length=hop_length
            )
            
            # Extract spectral flatness for each channel
            channel_features = []
            for ch in range(audio.channels):
                ch_data = audio.get_channel(ch)
                flatness = extractor.extract_spectral_flatness(ch_data)
                
                # Apply aggregation
                if aggregate == "mean":
                    result = np.mean(flatness, axis=1)
                elif aggregate == "std":
                    result = np.std(flatness, axis=1)
                elif aggregate == "mean_std":
                    result = np.concatenate([
                        np.mean(flatness, axis=1),
                        np.std(flatness, axis=1)
                    ])
                else:  # none
                    result = flatness.squeeze()
                
                channel_features.append(result)
            
            # Stack channels
            stacked = np.stack(channel_features, axis=0)
            
            return FeatureData(
                data=stacked.astype(np.float32),
                feature_type="spectral_flatness",
                sample_rate=audio.sample_rate,
                hop_length=hop_length,
                source_file=audio.file_path
            )
        
        result = extract_from_audio_or_list(audio_input, extract_flatness)
        self.set_output_data("feature", result)
        return True
