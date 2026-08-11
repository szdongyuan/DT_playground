# -*- coding: utf-8 -*-
"""
Pitch (F0) Feature Extraction Node

Extracts fundamental frequency (pitch) from audio signals.
Uses the pyin algorithm for robust pitch tracking.
"""

import numpy as np

from ....node_base import BaseNode, NodeCategory, register_node
from ....port import DataType
from ..base import FeatureData, extract_from_audio_or_list, validate_audio_input, AudioData
from src.ui.i18n import tr_
# Import unified feature extractor
from src.audio.features import FeatureExtractor


@register_node
class PitchNode(BaseNode):
    """
    基频节点
    
    提取音频的基频（音高/F0）轨迹。
    使用 pyin 算法进行稳健的基频追踪。
    """
    
    node_type = "pitch"
    display_name = tr_("Pitch (F0)")
    category = NodeCategory.FEATURE
    subcategory = tr_("1D features")
    subcategory_order = 10
    palette_order = 30
    description = tr_("Extract pitch (F0) trajectory")
    icon = "🎤"
    
    def _setup_ports(self):
        self.add_input("audio", DataType.AUDIO, tr_("Audio"))
        self.add_output("feature", DataType.FEATURE, tr_("Pitch trajectory"))
    
    def _setup_parameters(self):
        self.add_parameter(
            "fmin", "float", 65.0,
            display_name=tr_("Min frequency (Hz)"),
            min_value=20.0, max_value=500.0,
            description=tr_("Lowest detectable frequency (default C2 ~ 65 Hz)"),
        )
        self.add_parameter(
            "fmax", "float", 2093.0,
            display_name=tr_("Max frequency (Hz)"),
            min_value=200.0, max_value=8000.0,
            description=tr_("Highest detectable frequency (default C7 ~ 2093 Hz)"),
        )
        self.add_parameter(
            "hop_length", "int", 512,
            display_name=tr_("Hop length"),
            min_value=64, max_value=2048
        )
        self.add_parameter(
            "fill_unvoiced", "float", 0.0,
            display_name=tr_("Fill value for unvoiced frames"),
            min_value=0.0, max_value=1000.0,
            description=tr_("Use this value for frames where pitch cannot be detected"),
        )
        self.add_parameter(
            "normalize", "bool", False,
            display_name=tr_("Normalize"),
            description=tr_("Normalize pitch values to 0-1 range"),
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
        
        fmin = self.get_parameter("fmin")
        fmax = self.get_parameter("fmax")
        hop_length = self.get_parameter("hop_length")
        fill_unvoiced = self.get_parameter("fill_unvoiced")
        normalize = self.get_parameter("normalize")
        
        def extract_pitch(audio: AudioData) -> FeatureData:
            # Use unified FeatureExtractor
            extractor = FeatureExtractor(
                sample_rate=audio.sample_rate,
                hop_length=hop_length
            )
            
            # Extract pitch for each channel
            channel_features = []
            for ch in range(audio.channels):
                ch_data = audio.get_channel(ch)
                pitch = extractor.extract_pitch(
                    ch_data,
                    fmin=fmin,
                    fmax=fmax,
                    fill_na=fill_unvoiced
                )
                
                if normalize and pitch.max() > 0:
                    pitch = pitch / fmax  # Normalize by max frequency
                
                channel_features.append(pitch.squeeze())
            
            # Stack as (channels, frames)
            stacked = np.stack(channel_features, axis=0)
            
            return FeatureData(
                data=stacked.astype(np.float32),
                feature_type="pitch",
                sample_rate=audio.sample_rate,
                hop_length=hop_length,
                source_file=audio.file_path
            )
        
        result = extract_from_audio_or_list(audio_input, extract_pitch)
        self.set_output_data("feature", result)
        return True
