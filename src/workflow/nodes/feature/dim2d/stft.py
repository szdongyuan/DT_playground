# -*- coding: utf-8 -*-
"""
STFT Feature Extraction Node

Extracts Short-Time Fourier Transform spectrogram from audio signals.
"""

import librosa
import numpy as np

from ....node_base import BaseNode, NodeCategory, register_node
from ....port import DataType
from ..base import FeatureData, extract_from_audio_or_list, validate_audio_input, AudioData
from src.ui.i18n import tr_
# Import unified feature extractor
from src.audio.features import FeatureExtractor


@register_node
class STFTNode(BaseNode):
    """
    STFT节点
    
    提取短时傅里叶变换频谱。
    """
    
    node_type = "stft"
    display_name = "STFT"
    category = NodeCategory.FEATURE
    subcategory = tr_("2D features")
    subcategory_order = 20
    palette_order = 20
    description = tr_("Extract STFT spectrogram")
    icon = "🎼"
    
    def _setup_ports(self):
        self.add_input("audio", DataType.AUDIO, tr_("Audio"))
        self.add_output("feature", DataType.FEATURE, tr_("STFT spectrogram"))
    
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
            "output_type", "choice", "magnitude",
            display_name=tr_("Output type"),
            choices=["magnitude", "power", "db", "complex"]
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
        output_type = self.get_parameter("output_type")
        
        def extract_stft(audio: AudioData) -> FeatureData:
            # Use unified FeatureExtractor
            extractor = FeatureExtractor(
                sample_rate=audio.sample_rate,
                n_fft=n_fft,
                hop_length=hop_length
            )
            
            # Extract STFT for each channel
            channel_features = []
            for ch in range(audio.channels):
                ch_data = audio.get_channel(ch)
                
                if output_type == "db":
                    # Use FeatureExtractor's extract_stft (already converted to dB)
                    result = extractor.extract_stft(ch_data, to_db=True)
                elif output_type == "magnitude":
                    result = extractor.extract_stft(ch_data, to_db=False)
                elif output_type == "power":
                    mag = extractor.extract_stft(ch_data, to_db=False)
                    result = mag ** 2
                else:  # complex
                    # Use librosa directly for complex result
                    stft = librosa.stft(ch_data, n_fft=n_fft, hop_length=hop_length)
                    result = np.stack([stft.real, stft.imag], axis=0)
                
                channel_features.append(result)
            
            if output_type == "complex":
                # Complex mode: each channel is (2, freq, frames), merge as (channels*2, freq, frames)
                stacked = np.concatenate(channel_features, axis=0)
            else:
                # Stack as (channels, freq_bins, frames)
                stacked = np.stack(channel_features, axis=0)
            
            return FeatureData(
                data=stacked.astype(np.float32),
                feature_type="stft",
                sample_rate=audio.sample_rate,
                hop_length=hop_length,
                source_file=audio.file_path
            )
        
        result = extract_from_audio_or_list(audio_input, extract_stft)
        self.set_output_data("feature", result)
        return True
