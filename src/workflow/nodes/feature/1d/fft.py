# -*- coding: utf-8 -*-
"""
FFT Feature Extraction Node

Extracts global FFT spectrum from audio signals.
Unlike STFT, FFT performs a single Fourier transform on the entire signal,
producing a global spectrum representation without time dimension.
"""

import numpy as np

from ...node_base import BaseNode, NodeCategory, register_node
from ...port import DataType
from ..base import FeatureData, extract_from_audio_or_list, validate_audio_input, AudioData


@register_node
class FFTNode(BaseNode):
    """
    FFT节点
    
    提取快速傅里叶变换频谱（全局频谱）。
    
    与STFT不同，FFT对整个音频信号进行一次傅里叶变换，
    得到全局频谱表示，没有时间维度。
    """
    
    node_type = "fft"
    display_name = "FFT频谱"
    category = NodeCategory.FEATURE
    description = "提取FFT全局频谱"
    icon = "📶"
    
    def _setup_ports(self):
        self.add_input("audio", DataType.AUDIO, "音频")
        self.add_output("feature", DataType.FEATURE, "FFT频谱")
    
    def _setup_parameters(self):
        self.add_parameter(
            "n_fft", "choice", "auto",
            display_name="FFT点数",
            choices=["auto", "512", "1024", "2048", "4096", "8192", "16384"],
            description="auto: 使用信号长度作为FFT点数"
        )
        self.add_parameter(
            "output_type", "choice", "magnitude",
            display_name="输出类型",
            choices=["magnitude", "power", "db"]
        )
        self.add_parameter(
            "normalize", "bool", True,
            display_name="归一化",
            description="归一化到最大值"
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
        
        n_fft_param = self.get_parameter("n_fft")
        output_type = self.get_parameter("output_type")
        normalize = self.get_parameter("normalize")
        
        def extract_fft(audio: AudioData) -> FeatureData:
            # Extract FFT for each channel
            channel_features = []
            
            for ch in range(audio.channels):
                ch_data = audio.get_channel(ch)
                
                # Determine FFT size
                if n_fft_param == "auto":
                    # Use signal length as FFT size
                    n_fft = len(ch_data)
                else:
                    n_fft = int(n_fft_param)
                
                # Perform FFT (only positive frequencies)
                fft_result = np.fft.rfft(ch_data, n=n_fft)
                magnitude = np.abs(fft_result)
                
                if output_type == "magnitude":
                    result = magnitude
                elif output_type == "power":
                    result = magnitude ** 2
                else:  # db
                    result = 20 * np.log10(magnitude + 1e-10)
                
                if normalize and result.max() > 0:
                    if output_type == "db":
                        result = result - result.max()  # dB normalize to 0
                    else:
                        result = result / result.max()  # Linear normalize to 1
                
                channel_features.append(result)
            
            # Stack as (channels, freq_bins)
            stacked = np.stack(channel_features, axis=0)
            
            return FeatureData(
                data=stacked.astype(np.float32),
                feature_type="fft",
                sample_rate=audio.sample_rate,
                hop_length=0,  # FFT has no hop_length
                source_file=audio.file_path
            )
        
        result = extract_from_audio_or_list(audio_input, extract_fft)
        self.set_output_data("feature", result)
        return True
