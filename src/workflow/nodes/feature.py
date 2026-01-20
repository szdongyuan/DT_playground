# -*- coding: utf-8 -*-
"""
特征提取节点

提供各类音频特征提取功能：Mel频谱、MFCC、STFT、FFT等。
"""

import logging
from dataclasses import dataclass
from typing import List, Union

import librosa
import numpy as np

from ..node_base import BaseNode, NodeCategory, register_node
from ..port import DataType
from .data_source import AudioData
from .preprocessing import validate_audio_input


logger = logging.getLogger(__name__)


@dataclass
class FeatureData:
    """
    特征数据包装
    
    数据格式：
    - 多通道特征: (channels, features, frames)
    - 单通道特征: (1, features, frames)
    - 统计特征: (channels, features) 或 (1, features)
    """
    data: np.ndarray        # 特征数据
    feature_type: str       # 特征类型
    sample_rate: int        # 原始采样率
    hop_length: int         # 帧移
    source_file: str = ""   # 源文件路径
    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def channels(self) -> int:
        """返回通道数"""
        return self.data.shape[0]
    
    @property
    def is_mono(self) -> bool:
        """是否为单通道"""
        return self.channels == 1
    
    @property
    def duration(self) -> float:
        """估算时长（秒）"""
        if len(self.data.shape) >= 3:
            # (channels, features, frames)
            n_frames = self.data.shape[-1]
            return n_frames * self.hop_length / self.sample_rate
        elif len(self.data.shape) == 2:
            # 可能是旧格式 (features, frames) 或统计特征 (channels, features)
            n_frames = self.data.shape[-1]
            return n_frames * self.hop_length / self.sample_rate
        return 0.0
    
    def get_channel(self, channel_idx: int) -> np.ndarray:
        """获取指定通道的特征"""
        if channel_idx >= self.channels:
            raise ValueError(f"通道索引 {channel_idx} 超出范围 (共 {self.channels} 通道)")
        return self.data[channel_idx]


def extract_from_audio_or_list(audio_input, extract_func) -> Union[FeatureData, List[FeatureData]]:
    """
    从单个音频或音频列表提取特征
    """
    if isinstance(audio_input, list):
        return [extract_func(audio) for audio in audio_input]
    else:
        return extract_func(audio_input)


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
            # 对每个通道分别提取Mel频谱
            channel_features = []
            for ch in range(audio.channels):
                ch_data = audio.get_channel(ch)
                mel = librosa.feature.melspectrogram(
                    y=ch_data,
                    sr=audio.sample_rate,
                    n_mels=n_mels,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    fmin=fmin,
                    fmax=min(fmax, audio.sample_rate / 2)
                )
                
                if power_to_db:
                    mel = librosa.power_to_db(mel, ref=np.max)
                
                channel_features.append(mel)
            
            # 堆叠为 (channels, n_mels, frames)
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


@register_node
class MFCCNode(BaseNode):
    """
    MFCC节点
    
    提取梅尔频率倒谱系数。
    """
    
    node_type = "mfcc"
    display_name = "MFCC"
    category = NodeCategory.FEATURE
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
            # 对每个通道分别提取MFCC
            channel_features = []
            for ch in range(audio.channels):
                ch_data = audio.get_channel(ch)
                mfcc = librosa.feature.mfcc(
                    y=ch_data,
                    sr=audio.sample_rate,
                    n_mfcc=n_mfcc,
                    n_mels=n_mels,
                    n_fft=n_fft,
                    hop_length=hop_length
                )
                
                features = [mfcc]
                
                if include_delta:
                    delta = librosa.feature.delta(mfcc)
                    features.append(delta)
                
                if include_delta2:
                    delta2 = librosa.feature.delta(mfcc, order=2)
                    features.append(delta2)
                
                combined = np.vstack(features)
                channel_features.append(combined)
            
            # 堆叠为 (channels, n_features, frames)
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


@register_node
class STFTNode(BaseNode):
    """
    STFT节点
    
    提取短时傅里叶变换频谱。
    """
    
    node_type = "stft"
    display_name = "STFT"
    category = NodeCategory.FEATURE
    description = "提取STFT频谱"
    icon = "🎼"
    
    def _setup_ports(self):
        self.add_input("audio", DataType.AUDIO, "音频")
        self.add_output("feature", DataType.FEATURE, "STFT频谱")
    
    def _setup_parameters(self):
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
            "output_type", "choice", "magnitude",
            display_name="输出类型",
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
            # 对每个通道分别提取STFT
            channel_features = []
            for ch in range(audio.channels):
                ch_data = audio.get_channel(ch)
                stft = librosa.stft(
                    ch_data,
                    n_fft=n_fft,
                    hop_length=hop_length
                )
                
                if output_type == "magnitude":
                    result = np.abs(stft)
                elif output_type == "power":
                    result = np.abs(stft) ** 2
                elif output_type == "db":
                    result = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
                else:  # complex
                    # 分离实部和虚部，每个通道有2个子通道
                    result = np.stack([stft.real, stft.imag], axis=0)
                
                channel_features.append(result)
            
            if output_type == "complex":
                # complex模式：每通道已是(2, freq, frames)，合并为(channels*2, freq, frames)
                stacked = np.concatenate(channel_features, axis=0)
            else:
                # 堆叠为 (channels, freq_bins, frames)
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
            # 对每个通道分别提取FFT
            channel_features = []
            
            for ch in range(audio.channels):
                ch_data = audio.get_channel(ch)
                
                # 确定 FFT 点数
                if n_fft_param == "auto":
                    # 使用信号长度作为 FFT 点数
                    n_fft = len(ch_data)
                else:
                    n_fft = int(n_fft_param)
                
                # 执行 FFT（只取正频率部分）
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
                        result = result - result.max()  # dB 归一化到 0
                    else:
                        result = result / result.max()  # 线性归一化到 1
                
                channel_features.append(result)
            
            # 堆叠为 (channels, freq_bins)
            stacked = np.stack(channel_features, axis=0)
            
            return FeatureData(
                data=stacked.astype(np.float32),
                feature_type="fft",
                sample_rate=audio.sample_rate,
                hop_length=0,  # FFT 没有 hop_length
                source_file=audio.file_path
            )
        
        result = extract_from_audio_or_list(audio_input, extract_fft)
        self.set_output_data("feature", result)
        return True


@register_node
class StatisticsNode(BaseNode):
    """
    统计特征节点
    
    提取频谱统计特征（1D）。
    """
    
    node_type = "statistics"
    display_name = "统计特征"
    category = NodeCategory.FEATURE
    description = "提取频谱统计特征"
    icon = "📉"
    
    def _setup_ports(self):
        self.add_input("audio", DataType.AUDIO, "音频")
        self.add_output("feature", DataType.FEATURE, "统计特征")
    
    def _setup_parameters(self):
        self.add_parameter(
            "spectral_centroid", "bool", True,
            display_name="频谱质心"
        )
        self.add_parameter(
            "spectral_bandwidth", "bool", True,
            display_name="频谱带宽"
        )
        self.add_parameter(
            "spectral_rolloff", "bool", True,
            display_name="频谱滚降"
        )
        self.add_parameter(
            "zero_crossing_rate", "bool", True,
            display_name="过零率"
        )
        self.add_parameter(
            "rms", "bool", True,
            display_name="RMS能量"
        )
        self.add_parameter(
            "aggregate", "choice", "mean",
            display_name="聚合方式",
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
            """聚合单通道特征，feat 形状为 (1, frames)"""
            if aggregate == "mean":
                return np.mean(feat, axis=1)
            elif aggregate == "std":
                return np.std(feat, axis=1)
            elif aggregate == "mean_std":
                return np.concatenate([np.mean(feat, axis=1), np.std(feat, axis=1)])
            else:  # all - 返回完整特征
                return feat.flatten()
        
        def extract_stats(audio: AudioData) -> FeatureData:
            # 对每个通道分别提取统计特征
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
            
            # 堆叠为 (channels, n_features)
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

