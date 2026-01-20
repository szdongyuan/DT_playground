"""
音频特征提取器
支持多种特征类型及其组合
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class FeatureType(Enum):
    """特征类型枚举"""
    MEL_SPECTROGRAM = "mel_spectrogram"
    MFCC = "mfcc"
    MFCC_DELTA = "mfcc_delta"          # MFCC + 一阶差分
    MFCC_DELTA2 = "mfcc_delta2"        # MFCC + 一阶 + 二阶差分
    STFT = "stft"
    FFT = "fft"                        # 快速傅里叶变换（全局频谱）
    CQT = "cqt"                        # 常数Q变换
    CHROMA = "chroma"
    SPECTRAL_CONTRAST = "spectral_contrast"
    TONNETZ = "tonnetz"
    SPECTRAL_CENTROID = "spectral_centroid"
    SPECTRAL_BANDWIDTH = "spectral_bandwidth"
    SPECTRAL_ROLLOFF = "spectral_rolloff"
    ZERO_CROSSING_RATE = "zcr"
    RMS = "rms"
    RAW = "raw"
    
    @classmethod
    def get_display_name(cls, feature_type: str) -> str:
        """获取特征的显示名称"""
        display_names = {
            "mel_spectrogram": "Mel频谱图",
            "mfcc": "MFCC",
            "mfcc_delta": "MFCC + Δ",
            "mfcc_delta2": "MFCC + Δ + ΔΔ",
            "stft": "STFT",
            "fft": "FFT频谱",
            "cqt": "CQT频谱",
            "chroma": "色度特征",
            "spectral_contrast": "频谱对比度",
            "tonnetz": "调性网络",
            "spectral_centroid": "频谱质心",
            "spectral_bandwidth": "频谱带宽",
            "spectral_rolloff": "频谱滚降",
            "zcr": "过零率",
            "rms": "均方根能量",
            "raw": "原始波形",
        }
        return display_names.get(feature_type, feature_type)
    
    @classmethod
    def get_2d_features(cls) -> List[str]:
        """获取2D特征类型（适用于2D CNN）"""
        return ["mel_spectrogram", "mfcc", "mfcc_delta", "mfcc_delta2", 
                "stft", "cqt", "chroma", "spectral_contrast"]
    
    @classmethod
    def get_1d_features(cls) -> List[str]:
        """获取1D特征类型（适用于1D CNN或全连接）"""
        return ["fft", "spectral_centroid", "spectral_bandwidth", "spectral_rolloff", 
                "zcr", "rms", "tonnetz", "raw"]
    
    @classmethod
    def get_combinable_2d_features(cls) -> List[str]:
        """获取可组合的2D特征（通道堆叠）"""
        return ["mel_spectrogram", "mfcc", "cqt", "chroma", "spectral_contrast"]


class FeatureExtractor:
    """音频特征提取类"""
    
    def __init__(self, sample_rate: int = 44100,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 n_mels: int = 128,
                 n_mfcc: int = 20,
                 n_bins: int = 84,
                 bins_per_octave: int = 12):
        """
        初始化特征提取器
        
        Args:
            sample_rate: 采样率
            n_fft: FFT窗口大小
            hop_length: 帧移
            n_mels: Mel滤波器数量
            n_mfcc: MFCC系数数量
            n_bins: CQT频带数（默认84，即7个八度×12半音）
            bins_per_octave: 每八度频带数（默认12）
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
    
    def extract_mel_spectrogram(self, audio: np.ndarray,
                                 to_db: bool = True) -> np.ndarray:
        """
        提取Mel频谱图
        
        Args:
            audio: 音频数据
            to_db: 是否转换为分贝
            
        Returns:
            Mel频谱图 (n_mels, time_steps)
        """
        import librosa
        
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        if to_db:
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec
    
    def extract_mfcc(self, audio: np.ndarray,
                     delta: bool = False,
                     delta_delta: bool = False) -> np.ndarray:
        """
        提取MFCC特征
        
        Args:
            audio: 音频数据
            delta: 是否包含一阶差分
            delta_delta: 是否包含二阶差分
            
        Returns:
            MFCC特征 (n_mfcc * (1 + delta + delta_delta), time_steps)
        """
        import librosa
        
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        features = [mfcc]
        
        if delta:
            mfcc_delta = librosa.feature.delta(mfcc)
            features.append(mfcc_delta)
        
        if delta_delta:
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            features.append(mfcc_delta2)
        
        return np.vstack(features)
    
    def extract_stft(self, audio: np.ndarray,
                     to_db: bool = True) -> np.ndarray:
        """
        提取STFT频谱
        
        Args:
            audio: 音频数据
            to_db: 是否转换为分贝
            
        Returns:
            STFT频谱 (n_fft//2 + 1, time_steps)
        """
        import librosa
        
        stft = np.abs(librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        ))
        
        if to_db:
            stft = librosa.amplitude_to_db(stft, ref=np.max)
        
        return stft
    
    def extract_fft(self, audio: np.ndarray,
                    to_db: bool = True,
                    n_fft: int = None) -> np.ndarray:
        """
        提取FFT频谱（全局快速傅里叶变换）
        
        与STFT不同，FFT对整个音频信号进行一次傅里叶变换，
        得到全局频谱表示，没有时间维度。
        
        Args:
            audio: 音频数据
            to_db: 是否转换为分贝
            n_fft: FFT点数，默认使用音频长度
            
        Returns:
            FFT幅度谱 (n_fft//2 + 1,) - 只取正频率部分
        """
        if n_fft is None:
            # 使用音频长度作为 FFT 点数
            n_fft = len(audio)
        
        # 执行 FFT
        fft_result = np.fft.rfft(audio, n=n_fft)
        
        # 取幅度
        magnitude = np.abs(fft_result)
        
        if to_db:
            # 转换为分贝，避免log(0)
            magnitude = 20 * np.log10(magnitude + 1e-10)
            # 归一化到 0 为最大值
            magnitude = magnitude - magnitude.max()
        
        return magnitude
    
    def extract_cqt(self, audio: np.ndarray,
                    to_db: bool = True,
                    fmin: float = None) -> np.ndarray:
        """
        提取CQT频谱（常数Q变换）
        
        CQT的频率分辨率随频率呈对数变化，特别适合音乐信号分析。
        
        Args:
            audio: 音频数据
            to_db: 是否转换为分贝
            fmin: 最低频率，默认为C1 (约32.7Hz)
            
        Returns:
            CQT频谱 (n_bins, time_steps)
        """
        import librosa
        
        if fmin is None:
            fmin = librosa.note_to_hz('C1')  # 约32.7Hz
        
        cqt = np.abs(librosa.cqt(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            n_bins=self.n_bins,
            bins_per_octave=self.bins_per_octave,
            fmin=fmin
        ))
        
        if to_db:
            cqt = librosa.amplitude_to_db(cqt, ref=np.max)
        
        return cqt
    
    def extract_chroma(self, audio: np.ndarray) -> np.ndarray:
        """
        提取色度特征
        
        Returns:
            色度特征 (12, time_steps)
        """
        import librosa
        
        return librosa.feature.chroma_stft(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
    
    def extract_spectral_contrast(self, audio: np.ndarray) -> np.ndarray:
        """
        提取频谱对比度
        
        Returns:
            频谱对比度 (7, time_steps)
        """
        import librosa
        
        return librosa.feature.spectral_contrast(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
    
    def extract_tonnetz(self, audio: np.ndarray) -> np.ndarray:
        """
        提取调性网络特征
        
        Returns:
            调性网络 (6, time_steps)
        """
        import librosa
        
        return librosa.feature.tonnetz(
            y=audio,
            sr=self.sample_rate
        )
    
    def extract_spectral_centroid(self, audio: np.ndarray) -> np.ndarray:
        """提取频谱质心"""
        import librosa
        return librosa.feature.spectral_centroid(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )
    
    def extract_spectral_bandwidth(self, audio: np.ndarray) -> np.ndarray:
        """提取频谱带宽"""
        import librosa
        return librosa.feature.spectral_bandwidth(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )
    
    def extract_spectral_rolloff(self, audio: np.ndarray) -> np.ndarray:
        """提取频谱滚降"""
        import librosa
        return librosa.feature.spectral_rolloff(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )
    
    def extract_zero_crossing_rate(self, audio: np.ndarray) -> np.ndarray:
        """提取过零率"""
        import librosa
        return librosa.feature.zero_crossing_rate(
            audio, hop_length=self.hop_length
        )
    
    def extract_rms(self, audio: np.ndarray) -> np.ndarray:
        """提取均方根能量"""
        import librosa
        return librosa.feature.rms(
            y=audio, hop_length=self.hop_length
        )
    
    def extract_spectral_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        提取频谱特征集合
        
        Returns:
            包含多种频谱特征的字典
        """
        return {
            'spectral_centroid': self.extract_spectral_centroid(audio)[0],
            'spectral_bandwidth': self.extract_spectral_bandwidth(audio)[0],
            'spectral_rolloff': self.extract_spectral_rolloff(audio)[0],
            'zero_crossing_rate': self.extract_zero_crossing_rate(audio)[0]
        }
    
    def extract_feature(self, audio: np.ndarray, feature_type: str, 
                        normalize: bool = True) -> np.ndarray:
        """
        提取单一类型特征
        
        Args:
            audio: 音频数据
            feature_type: 特征类型
            normalize: 是否归一化
            
        Returns:
            特征数组
        """
        if feature_type == "mel_spectrogram":
            features = self.extract_mel_spectrogram(audio)
        elif feature_type == "mfcc":
            features = self.extract_mfcc(audio)
        elif feature_type == "mfcc_delta":
            features = self.extract_mfcc(audio, delta=True)
        elif feature_type == "mfcc_delta2":
            features = self.extract_mfcc(audio, delta=True, delta_delta=True)
        elif feature_type == "stft":
            features = self.extract_stft(audio)
        elif feature_type == "fft":
            features = self.extract_fft(audio)
        elif feature_type == "cqt":
            features = self.extract_cqt(audio)
        elif feature_type == "chroma":
            features = self.extract_chroma(audio)
        elif feature_type == "spectral_contrast":
            features = self.extract_spectral_contrast(audio)
        elif feature_type == "tonnetz":
            features = self.extract_tonnetz(audio)
        elif feature_type == "spectral_centroid":
            features = self.extract_spectral_centroid(audio)
        elif feature_type == "spectral_bandwidth":
            features = self.extract_spectral_bandwidth(audio)
        elif feature_type == "spectral_rolloff":
            features = self.extract_spectral_rolloff(audio)
        elif feature_type == "zcr":
            features = self.extract_zero_crossing_rate(audio)
        elif feature_type == "rms":
            features = self.extract_rms(audio)
        elif feature_type == "raw":
            features = audio.reshape(1, -1)  # (1, samples)
        else:
            # 默认使用Mel频谱图
            features = self.extract_mel_spectrogram(audio)
        
        if normalize:
            features = self._normalize(features)
        
        return features
    
    def extract_combined_features(self, audio: np.ndarray, 
                                   feature_types: List[str],
                                   normalize: bool = True,
                                   combine_mode: str = "channel") -> np.ndarray:
        """
        提取并组合多种特征
        
        Args:
            audio: 音频数据
            feature_types: 特征类型列表
            normalize: 是否归一化
            combine_mode: 组合模式
                - "channel": 通道堆叠 (适用于2D CNN, 输出 H x W x C)
                - "concat": 纵向拼接 (适用于频率轴维度)
                - "flatten": 展平后拼接 (适用于1D特征)
                
        Returns:
            组合后的特征数组
        """
        if not feature_types:
            feature_types = ["mel_spectrogram"]
        
        if len(feature_types) == 1:
            features = self.extract_feature(audio, feature_types[0], normalize)
            # 单一特征，添加通道维度
            if features.ndim == 2:
                features = features[..., np.newaxis]
            return features
        
        # 提取所有特征
        all_features = []
        for ft in feature_types:
            feat = self.extract_feature(audio, ft, normalize)
            all_features.append(feat)
        
        if combine_mode == "channel":
            # 通道堆叠模式: 统一时间维度后作为不同通道
            return self._combine_as_channels(all_features)
        elif combine_mode == "concat":
            # 纵向拼接模式
            return self._combine_concat(all_features)
        elif combine_mode == "flatten":
            # 展平拼接模式
            return self._combine_flatten(all_features)
        else:
            return self._combine_as_channels(all_features)
    
    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """归一化特征"""
        mean = features.mean()
        std = features.std()
        if std > 0:
            return (features - mean) / std
        return features - mean
    
    def _combine_as_channels(self, features_list: List[np.ndarray]) -> np.ndarray:
        """
        将多个特征作为不同通道组合
        统一到相同的时间维度，不同特征作为不同通道
        
        Returns:
            (max_freq_bins, time_steps, num_features) 的数组
        """
        # 找到最大的频率维度和统一的时间维度
        max_freq = 0
        min_time = float('inf')
        
        processed = []
        for feat in features_list:
            if feat.ndim == 1:
                feat = feat.reshape(1, -1)
            max_freq = max(max_freq, feat.shape[0])
            min_time = min(min_time, feat.shape[1])
            processed.append(feat)
        
        min_time = int(min_time)
        
        # 统一尺寸并堆叠
        channels = []
        for feat in processed:
            # 裁剪时间维度
            feat = feat[:, :min_time]
            # 填充频率维度（如果需要）
            if feat.shape[0] < max_freq:
                pad_size = max_freq - feat.shape[0]
                feat = np.pad(feat, ((0, pad_size), (0, 0)), mode='constant')
            channels.append(feat)
        
        # 堆叠为 (freq, time, channels)
        return np.stack(channels, axis=-1)
    
    def _combine_concat(self, features_list: List[np.ndarray]) -> np.ndarray:
        """纵向拼接特征（沿频率轴）"""
        min_time = min(f.shape[-1] if f.ndim > 1 else len(f) for f in features_list)
        
        processed = []
        for feat in features_list:
            if feat.ndim == 1:
                feat = feat.reshape(1, -1)
            feat = feat[:, :min_time]
            processed.append(feat)
        
        combined = np.vstack(processed)
        return combined[..., np.newaxis]
    
    def _combine_flatten(self, features_list: List[np.ndarray]) -> np.ndarray:
        """展平后拼接所有特征"""
        flattened = []
        for feat in features_list:
            flattened.append(feat.flatten())
        return np.concatenate(flattened)
    
    def get_feature_shape(self, audio_length: int, feature_types: List[str],
                          combine_mode: str = "channel") -> Tuple[int, ...]:
        """
        计算特征输出形状
        
        Args:
            audio_length: 音频样本数
            feature_types: 特征类型列表
            combine_mode: 组合模式
            
        Returns:
            特征形状元组
        """
        time_steps = int((audio_length - self.n_fft) / self.hop_length) + 1
        
        if len(feature_types) == 1:
            ft = feature_types[0]
            if ft == "mel_spectrogram":
                return (self.n_mels, time_steps, 1)
            elif ft == "mfcc":
                return (self.n_mfcc, time_steps, 1)
            elif ft == "mfcc_delta":
                return (self.n_mfcc * 2, time_steps, 1)
            elif ft == "mfcc_delta2":
                return (self.n_mfcc * 3, time_steps, 1)
            elif ft == "stft":
                return (self.n_fft // 2 + 1, time_steps, 1)
            elif ft == "fft":
                # FFT 输出长度为 n_fft//2 + 1，n_fft 等于信号长度
                return (audio_length // 2 + 1, 1)
            elif ft == "cqt":
                return (self.n_bins, time_steps, 1)
            elif ft == "chroma":
                return (12, time_steps, 1)
            elif ft == "spectral_contrast":
                return (7, time_steps, 1)
            elif ft == "tonnetz":
                return (6, time_steps, 1)
            elif ft in ["spectral_centroid", "spectral_bandwidth", 
                        "spectral_rolloff", "zcr", "rms"]:
                return (1, time_steps, 1)
            elif ft == "raw":
                return (audio_length, 1)
            else:
                return (self.n_mels, time_steps, 1)
        
        # 多特征组合
        if combine_mode == "channel":
            # 计算最大频率维度
            max_freq = 0
            for ft in feature_types:
                shape = self.get_feature_shape(audio_length, [ft], "channel")
                max_freq = max(max_freq, shape[0])
            return (max_freq, time_steps, len(feature_types))
        elif combine_mode == "concat":
            total_freq = 0
            for ft in feature_types:
                shape = self.get_feature_shape(audio_length, [ft], "channel")
                total_freq += shape[0]
            return (total_freq, time_steps, 1)
        else:
            # flatten模式，返回1D
            total_size = 0
            for ft in feature_types:
                shape = self.get_feature_shape(audio_length, [ft], "channel")
                total_size += np.prod(shape)
            return (int(total_size),)
    
    def extract_all(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """提取所有特征"""
        return {
            'mel_spectrogram': self.extract_mel_spectrogram(audio),
            'mfcc': self.extract_mfcc(audio, delta=True, delta_delta=True),
            'stft': self.extract_stft(audio),
            'fft': self.extract_fft(audio),
            'cqt': self.extract_cqt(audio),
            'chroma': self.extract_chroma(audio),
            'spectral_contrast': self.extract_spectral_contrast(audio),
            'tonnetz': self.extract_tonnetz(audio),
            **self.extract_spectral_features(audio)
        }


# 便捷函数
def get_available_features() -> List[Dict[str, str]]:
    """获取所有可用的特征类型"""
    features = []
    for ft in FeatureType:
        if ft != FeatureType.RAW:  # raw 特殊处理
            features.append({
                "id": ft.value,
                "name": FeatureType.get_display_name(ft.value),
                "is_2d": ft.value in FeatureType.get_2d_features()
            })
    features.append({
        "id": "raw",
        "name": "原始波形",
        "is_2d": False
    })
    return features


def get_feature_categories() -> Dict[str, List[str]]:
    """获取特征分类"""
    return {
        "频谱特征 (2D)": ["mel_spectrogram", "stft", "cqt", "chroma", "spectral_contrast"],
        "MFCC系列 (2D)": ["mfcc", "mfcc_delta", "mfcc_delta2"],
        "音调特征": ["tonnetz"],
        "频域特征 (1D)": ["fft"],
        "统计特征 (1D)": ["spectral_centroid", "spectral_bandwidth", 
                         "spectral_rolloff", "zcr", "rms"],
        "原始数据": ["raw"]
    }
