"""
Audio Preprocessor
"""

from typing import Optional, Tuple

import numpy as np


class AudioPreprocessor:
    """音频预处理类"""
    
    def __init__(self, sample_rate: int = 44100):
        """
        初始化预处理器
        
        Args:
            sample_rate: 采样率
        """
        self.sample_rate = sample_rate
    
    def normalize(self, audio: np.ndarray, 
                  method: str = 'peak') -> np.ndarray:
        """
        音频归一化
        
        Args:
            audio: 音频数据
            method: 归一化方法 ('peak', 'rms', 'minmax')
            
        Returns:
            归一化后的音频
        """
        if method == 'peak':
            # 峰值归一化到[-1, 1]
            peak = np.max(np.abs(audio))
            if peak > 0:
                return audio / peak
            return audio
            
        elif method == 'rms':
            # RMS归一化
            rms = np.sqrt(np.mean(audio ** 2))
            if rms > 0:
                return audio / rms
            return audio
            
        elif method == 'minmax':
            # Min-Max归一化到[0, 1]
            min_val = np.min(audio)
            max_val = np.max(audio)
            if max_val - min_val > 0:
                return (audio - min_val) / (max_val - min_val)
            return audio
            
        else:
            raise ValueError(f"未知的归一化方法: {method}")
    
    def pad_or_trim(self, audio: np.ndarray, 
                    target_length: int,
                    mode: str = 'constant') -> np.ndarray:
        """
        填充或裁剪音频到目标长度
        
        Args:
            audio: 音频数据
            target_length: 目标长度（采样点数）
            mode: 填充模式
            
        Returns:
            处理后的音频
        """
        current_length = len(audio)
        
        if current_length == target_length:
            return audio
        elif current_length > target_length:
            # 裁剪
            return audio[:target_length]
        else:
            # 填充
            padding = target_length - current_length
            if mode == 'constant':
                return np.pad(audio, (0, padding), mode='constant', constant_values=0)
            elif mode == 'wrap':
                return np.pad(audio, (0, padding), mode='wrap')
            elif mode == 'reflect':
                return np.pad(audio, (0, padding), mode='reflect')
            else:
                return np.pad(audio, (0, padding), mode='constant', constant_values=0)
    
    def resample(self, audio: np.ndarray, 
                 orig_sr: int, 
                 target_sr: int) -> np.ndarray:
        """
        重采样
        
        Args:
            audio: 音频数据
            orig_sr: 原始采样率
            target_sr: 目标采样率
            
        Returns:
            重采样后的音频
        """
        if orig_sr == target_sr:
            return audio
        
        try:
            import librosa
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            import resampy
            return resampy.resample(audio, orig_sr, target_sr)
    
    def remove_silence(self, audio: np.ndarray,
                       top_db: float = 20,
                       frame_length: int = 2048,
                       hop_length: int = 512) -> np.ndarray:
        """
        移除静音段
        
        Args:
            audio: 音频数据
            top_db: 低于参考的dB阈值
            frame_length: 帧长度
            hop_length: 帧移
            
        Returns:
            移除静音后的音频
        """
        try:
            import librosa
            
            # 找出非静音区间
            intervals = librosa.effects.split(
                audio, 
                top_db=top_db,
                frame_length=frame_length,
                hop_length=hop_length
            )
            
            # 合并非静音段
            if len(intervals) == 0:
                return audio
            
            non_silent = np.concatenate([audio[start:end] for start, end in intervals])
            return non_silent
            
        except Exception as e:
            print(f"移除静音失败: {e}")
            return audio
    
    def apply_preemphasis(self, audio: np.ndarray, 
                          coef: float = 0.97) -> np.ndarray:
        """
        预加重滤波
        
        Args:
            audio: 音频数据
            coef: 预加重系数
            
        Returns:
            预加重后的音频
        """
        return np.append(audio[0], audio[1:] - coef * audio[:-1])
    
    def split_frames(self, audio: np.ndarray,
                     frame_length: int = 2048,
                     hop_length: int = 512) -> np.ndarray:
        """
        分帧
        
        Args:
            audio: 音频数据
            frame_length: 帧长度
            hop_length: 帧移
            
        Returns:
            帧矩阵 (n_frames, frame_length)
        """
        try:
            import librosa
            return librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length).T
        except Exception:
            # 手动分帧
            n_frames = 1 + (len(audio) - frame_length) // hop_length
            frames = np.zeros((n_frames, frame_length))
            for i in range(n_frames):
                start = i * hop_length
                frames[i] = audio[start:start + frame_length]
            return frames
    
    def apply_window(self, frames: np.ndarray, 
                     window: str = 'hann') -> np.ndarray:
        """
        应用窗函数
        
        Args:
            frames: 帧矩阵
            window: 窗函数类型
            
        Returns:
            加窗后的帧
        """
        from scipy.signal import get_window
        
        frame_length = frames.shape[1]
        win = get_window(window, frame_length)
        
        return frames * win

