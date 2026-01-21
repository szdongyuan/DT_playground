"""
Audio Loader
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class AudioData:
    """音频数据类"""
    data: np.ndarray          # 音频数据
    sample_rate: int          # 采样率
    duration: float           # 时长（秒）
    channels: int             # 声道数
    file_path: Optional[str]  # 文件路径


class AudioLoader:
    """音频加载器类"""
    
    # 支持的音频格式
    SUPPORTED_FORMATS = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac']
    
    def __init__(self, target_sr: int = 44100, mono: bool = True):
        """
        初始化音频加载器
        
        Args:
            target_sr: 目标采样率
            mono: 是否转换为单声道
        """
        self.target_sr = target_sr
        self.mono = mono
    
    def load(self, file_path: str, 
             duration: Optional[float] = None,
             offset: float = 0.0) -> AudioData:
        """
        加载音频文件
        
        Args:
            file_path: 音频文件路径
            duration: 加载时长（秒），None表示加载全部
            offset: 起始偏移（秒）
            
        Returns:
            AudioData对象
        """
        try:
            import librosa
            
            # 检查文件是否存在
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"文件不存在: {file_path}")
            
            # 检查格式是否支持
            ext = os.path.splitext(file_path)[1].lower()
            if ext not in self.SUPPORTED_FORMATS:
                raise ValueError(f"不支持的音频格式: {ext}")
            
            # 加载音频
            data, sr = librosa.load(
                file_path,
                sr=self.target_sr,
                mono=self.mono,
                duration=duration,
                offset=offset
            )
            
            # 计算时长
            audio_duration = len(data) / sr
            
            # 确定声道数
            channels = 1 if self.mono else (data.ndim if data.ndim > 1 else 1)
            
            return AudioData(
                data=data,
                sample_rate=sr,
                duration=audio_duration,
                channels=channels,
                file_path=file_path
            )
            
        except Exception as e:
            raise RuntimeError(f"加载音频失败: {e}")
    
    def load_batch(self, file_paths: List[str], 
                   max_duration: Optional[float] = None) -> List[AudioData]:
        """
        批量加载音频文件
        
        Args:
            file_paths: 音频文件路径列表
            max_duration: 最大时长（秒）
            
        Returns:
            AudioData对象列表
        """
        results = []
        for path in file_paths:
            try:
                audio = self.load(path, duration=max_duration)
                results.append(audio)
            except Exception as e:
                print(f"跳过文件 {path}: {e}")
        return results
    
    def get_info(self, file_path: str) -> dict:
        """
        获取音频文件信息（不加载完整数据）
        
        Args:
            file_path: 音频文件路径
            
        Returns:
            包含音频信息的字典
        """
        try:
            import librosa
            
            duration = librosa.get_duration(path=file_path)
            
            # 加载少量数据获取采样率
            _, sr = librosa.load(file_path, sr=None, duration=0.1)
            
            return {
                'duration': duration,
                'sample_rate': sr,
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'format': os.path.splitext(file_path)[1].lower()
            }
            
        except Exception as e:
            raise RuntimeError(f"获取音频信息失败: {e}")
    
    @staticmethod
    def is_supported(file_path: str) -> bool:
        """检查文件格式是否支持"""
        ext = os.path.splitext(file_path)[1].lower()
        return ext in AudioLoader.SUPPORTED_FORMATS

