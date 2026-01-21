"""
Audio Data Augmentation
"""

from typing import Optional, Tuple

import numpy as np


class AudioAugmenter:
    """音频数据增强类"""
    
    def __init__(self, sample_rate: int = 44100):
        """初始化增强器"""
        self.sample_rate = sample_rate
    
    def time_stretch(self, audio: np.ndarray, rate: float = 1.0) -> np.ndarray:
        """时间拉伸/压缩"""
        import librosa
        return librosa.effects.time_stretch(audio, rate=rate)
    
    def pitch_shift(self, audio: np.ndarray, n_steps: float = 0) -> np.ndarray:
        """音高偏移"""
        import librosa
        return librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=n_steps)
    
    def add_noise(self, audio: np.ndarray, noise_level: float = 0.005) -> np.ndarray:
        """添加高斯噪声"""
        noise = np.random.randn(len(audio)) * noise_level
        return audio + noise
    
    def time_shift(self, audio: np.ndarray, shift_max: float = 0.2) -> np.ndarray:
        """时间偏移"""
        shift = int(len(audio) * np.random.uniform(-shift_max, shift_max))
        return np.roll(audio, shift)
    
    def volume_change(self, audio: np.ndarray, gain_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """音量变化"""
        gain = np.random.uniform(*gain_range)
        return audio * gain
    
    def random_augment(self, audio: np.ndarray) -> np.ndarray:
        """随机应用多种增强"""
        augmented = audio.copy()
        
        if np.random.random() > 0.5:
            augmented = self.time_stretch(augmented, np.random.uniform(0.8, 1.2))
        
        if np.random.random() > 0.5:
            augmented = self.pitch_shift(augmented, np.random.uniform(-2, 2))
        
        if np.random.random() > 0.5:
            augmented = self.add_noise(augmented, np.random.uniform(0.001, 0.01))
        
        if np.random.random() > 0.5:
            augmented = self.volume_change(augmented)
        
        return augmented

