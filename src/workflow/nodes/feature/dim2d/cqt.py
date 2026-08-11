# -*- coding: utf-8 -*-
"""
CQT Feature Extraction Node

Extracts Constant-Q Transform spectrogram from audio signals.
CQT provides logarithmically-spaced frequency resolution, making it
particularly suitable for music signal analysis.
"""

import numpy as np

from ....node_base import BaseNode, NodeCategory, register_node
from ....port import DataType
from ..base import FeatureData, extract_from_audio_or_list, validate_audio_input, AudioData
from src.ui.i18n import tr_
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
    display_name = tr_("CQT spectrum")
    category = NodeCategory.FEATURE
    subcategory = tr_("2D features")
    subcategory_order = 20
    palette_order = 40
    description = tr_("Extract Constant-Q Transform (CQT) spectrogram")
    icon = "🎵"
    
    def _setup_ports(self):
        self.add_input("audio", DataType.AUDIO, tr_("Audio"))
        self.add_output("feature", DataType.FEATURE, tr_("CQT spectrum"))
    
    def _setup_parameters(self):
        self.add_parameter(
            "n_bins", "int", 84,
            display_name=tr_("Bins"),
            min_value=12, max_value=168,
            description=tr_("Number of CQT bins (default 84 = 7 octaves × 12 semitones)"),
        )
        self.add_parameter(
            "bins_per_octave", "int", 12,
            display_name=tr_("Bins per octave"),
            min_value=6, max_value=48,
            description=tr_("Bins per octave (default 12 = semitone resolution)"),
        )
        self.add_parameter(
            "hop_length", "int", 512,
            display_name=tr_("Hop length"),
            min_value=64, max_value=2048
        )
        self.add_parameter(
            "fmin", "float", 32.7,
            display_name=tr_("Min frequency (Hz)"),
            min_value=10.0, max_value=500.0,
            description=tr_("Minimum frequency (default C1 ~ 32.7 Hz)"),
        )
        self.add_parameter(
            "to_db", "bool", True,
            display_name=tr_("Convert to dB"),
            description=tr_("Convert magnitude to dB scale"),
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
