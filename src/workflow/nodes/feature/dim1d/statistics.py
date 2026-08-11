# -*- coding: utf-8 -*-
"""
Statistics Feature Extraction Node

Extracts spectral statistical features (1D) from audio signals.
"""

import librosa
import numpy as np

from ....node_base import BaseNode, NodeCategory, register_node
from ....port import DataType
from ..base import FeatureData, extract_from_audio_or_list, validate_audio_input, AudioData
from src.ui.i18n import tr_
@register_node
class StatisticsNode(BaseNode):
    """
    统计特征节点
    
    提取频谱统计特征（1D）。
    """
    
    node_type = "statistics"
    display_name = tr_("Statistical features")
    category = NodeCategory.FEATURE
    subcategory = tr_("1D features")
    subcategory_order = 10
    palette_order = 20
    description = tr_("Extract statistical spectral features")
    icon = "📉"
    
    def _setup_ports(self):
        self.add_input("audio", DataType.AUDIO, tr_("Audio"))
        self.add_output("feature", DataType.FEATURE, tr_("Statistical features"))
    
    def _setup_parameters(self):
        self.add_parameter(
            "spectral_centroid", "bool", True,
            display_name=tr_("Spectral centroid")
        )
        self.add_parameter(
            "spectral_bandwidth", "bool", True,
            display_name=tr_("Spectral bandwidth")
        )
        self.add_parameter(
            "spectral_rolloff", "bool", True,
            display_name=tr_("Spectral rolloff")
        )
        self.add_parameter(
            "zero_crossing_rate", "bool", True,
            display_name=tr_("Zero-crossing rate")
        )
        self.add_parameter(
            "rms", "bool", True,
            display_name=tr_("RMS energy")
        )
        self.add_parameter(
            "aggregate", "choice", "mean",
            display_name=tr_("Aggregation"),
            choices=["mean", "std", "mean_std", "all"]
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
        
        aggregate = self.get_parameter("aggregate")
        
        def aggregate_feature(feat: np.ndarray) -> np.ndarray:
            """Aggregate single channel feature, feat shape is (1, frames)"""
            if aggregate == "mean":
                return np.mean(feat, axis=1)
            elif aggregate == "std":
                return np.std(feat, axis=1)
            elif aggregate == "mean_std":
                return np.concatenate([np.mean(feat, axis=1), np.std(feat, axis=1)])
            else:  # all - return complete feature
                return feat.flatten()
        
        def extract_stats(audio: AudioData) -> FeatureData:
            # Extract statistics for each channel
            channel_features = []
            
            for ch in range(audio.channels):
                ch_data = audio.get_channel(ch)
                features = []
                
                if self.get_parameter("spectral_centroid"):
                    feat = librosa.feature.spectral_centroid(y=ch_data, sr=audio.sample_rate)
                    features.append(aggregate_feature(feat))
                
                if self.get_parameter("spectral_bandwidth"):
                    feat = librosa.feature.spectral_bandwidth(y=ch_data, sr=audio.sample_rate)
                    features.append(aggregate_feature(feat))
                
                if self.get_parameter("spectral_rolloff"):
                    feat = librosa.feature.spectral_rolloff(y=ch_data, sr=audio.sample_rate)
                    features.append(aggregate_feature(feat))
                
                if self.get_parameter("zero_crossing_rate"):
                    feat = librosa.feature.zero_crossing_rate(ch_data)
                    features.append(aggregate_feature(feat))
                
                if self.get_parameter("rms"):
                    feat = librosa.feature.rms(y=ch_data)
                    features.append(aggregate_feature(feat))
                
                if not features:
                    features.append(np.zeros(1))
                
                combined = np.concatenate(features)
                channel_features.append(combined)
            
            # Stack as (channels, n_features)
            stacked = np.stack(channel_features, axis=0)
            
            return FeatureData(
                data=stacked.astype(np.float32),
                feature_type="statistics",
                sample_rate=audio.sample_rate,
                hop_length=512,
                source_file=audio.file_path
            )
        
        result = extract_from_audio_or_list(audio_input, extract_stats)
        self.set_output_data("feature", result)
        return True
