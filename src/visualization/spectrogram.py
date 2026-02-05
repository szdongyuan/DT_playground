"""
Spectrogram Visualization Tools
"""

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from src.ui.i18n import tr_
class SpectrogramPlotter:
    """频谱图绘制器"""
    
    def __init__(self, figsize: Tuple[int, int] = (10, 4), dpi: int = 100):
        """
        初始化频谱图绘制器
        
        Args:
            figsize: 图像尺寸
            dpi: 分辨率
        """
        self.figsize = figsize
        self.dpi = dpi
        self.fig = None
        self.ax = None
    
    def plot_mel_spectrogram(self, audio: np.ndarray,
                              sample_rate: int,
                              n_mels: int = 128,
                              n_fft: int = 2048,
                              hop_length: int = 512,
                              fmax: int = 8000,
                              cmap: str = 'viridis') -> Figure:
        """
        绘制Mel频谱图
        
        Args:
            audio: 音频数据
            sample_rate: 采样率
            n_mels: Mel滤波器数量
            n_fft: FFT窗口大小
            hop_length: 帧移
            fmax: 最大频率
            cmap: 颜色映射
            
        Returns:
            matplotlib Figure对象
        """
        import librosa
        import librosa.display
        
        self.fig, self.ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # 计算Mel频谱图
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sample_rate,
            n_mels=n_mels, n_fft=n_fft,
            hop_length=hop_length, fmax=fmax
        )
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # 绘制
        img = librosa.display.specshow(
            mel_db, sr=sample_rate, hop_length=hop_length,
            x_axis='time', y_axis='mel',
            ax=self.ax, cmap=cmap
        )
        
        # 颜色条
        cbar = self.fig.colorbar(img, ax=self.ax, format='%+2.0f dB')
        cbar.ax.yaxis.set_tick_params(color='#cdd6f4')
        cbar.outline.set_edgecolor('#45475a')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#a6adc8')
        
        self._apply_style(tr_("Mel spectrogram"))
        
        return self.fig
    
    def plot_stft(self, audio: np.ndarray,
                  sample_rate: int,
                  n_fft: int = 2048,
                  hop_length: int = 512,
                  cmap: str = 'viridis') -> Figure:
        """
        绘制STFT频谱图
        
        Args:
            audio: 音频数据
            sample_rate: 采样率
            n_fft: FFT窗口大小
            hop_length: 帧移
            cmap: 颜色映射
            
        Returns:
            matplotlib Figure对象
        """
        import librosa
        import librosa.display
        
        self.fig, self.ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # 计算STFT
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        
        # 绘制
        img = librosa.display.specshow(
            stft_db, sr=sample_rate, hop_length=hop_length,
            x_axis='time', y_axis='hz',
            ax=self.ax, cmap=cmap
        )
        
        cbar = self.fig.colorbar(img, ax=self.ax, format='%+2.0f dB')
        cbar.ax.yaxis.set_tick_params(color='#cdd6f4')
        cbar.outline.set_edgecolor('#45475a')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#a6adc8')
        
        self._apply_style(tr_("STFT spectrogram"))
        
        return self.fig
    
    def plot_mfcc(self, audio: np.ndarray,
                  sample_rate: int,
                  n_mfcc: int = 20,
                  n_fft: int = 2048,
                  hop_length: int = 512,
                  cmap: str = 'viridis') -> Figure:
        """
        绘制MFCC
        
        Args:
            audio: 音频数据
            sample_rate: 采样率
            n_mfcc: MFCC系数数量
            n_fft: FFT窗口大小
            hop_length: 帧移
            cmap: 颜色映射
            
        Returns:
            matplotlib Figure对象
        """
        import librosa
        import librosa.display
        
        self.fig, self.ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # 计算MFCC
        mfcc = librosa.feature.mfcc(
            y=audio, sr=sample_rate,
            n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
        )
        
        # 绘制
        img = librosa.display.specshow(
            mfcc, sr=sample_rate, hop_length=hop_length,
            x_axis='time', ax=self.ax, cmap=cmap
        )
        
        cbar = self.fig.colorbar(img, ax=self.ax)
        cbar.ax.yaxis.set_tick_params(color='#cdd6f4')
        cbar.outline.set_edgecolor('#45475a')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#a6adc8')
        
        self.ax.set_ylabel(tr_("MFCC coefficients"), color='#cdd6f4')
        self._apply_style(tr_("MFCC"))
        
        return self.fig
    
    def plot_chroma(self, audio: np.ndarray,
                    sample_rate: int,
                    hop_length: int = 512,
                    cmap: str = 'viridis') -> Figure:
        """
        绘制色度图
        
        Args:
            audio: 音频数据
            sample_rate: 采样率
            hop_length: 帧移
            cmap: 颜色映射
            
        Returns:
            matplotlib Figure对象
        """
        import librosa
        import librosa.display
        
        self.fig, self.ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # 计算色度特征
        chroma = librosa.feature.chroma_stft(
            y=audio, sr=sample_rate, hop_length=hop_length
        )
        
        # 绘制
        img = librosa.display.specshow(
            chroma, sr=sample_rate, hop_length=hop_length,
            x_axis='time', y_axis='chroma',
            ax=self.ax, cmap=cmap
        )
        
        cbar = self.fig.colorbar(img, ax=self.ax)
        cbar.ax.yaxis.set_tick_params(color='#cdd6f4')
        cbar.outline.set_edgecolor('#45475a')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#a6adc8')
        
        self._apply_style(tr_("Chroma"))
        
        return self.fig
    
    def plot_comparison(self, audio: np.ndarray,
                        sample_rate: int,
                        n_fft: int = 2048,
                        hop_length: int = 512,
                        cmap: str = 'viridis') -> Figure:
        """
        绘制多特征对比图
        
        Args:
            audio: 音频数据
            sample_rate: 采样率
            n_fft: FFT窗口大小
            hop_length: 帧移
            cmap: 颜色映射
            
        Returns:
            matplotlib Figure对象
        """
        import librosa
        import librosa.display
        
        self.fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=self.dpi)
        
        # Mel频谱图
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        librosa.display.specshow(mel_db, sr=sample_rate, hop_length=hop_length,
                                  x_axis='time', y_axis='mel', ax=axes[0, 0], cmap=cmap)
        axes[0, 0].set_title(tr_("Mel spectrogram"), color='#cdd6f4')
        
        # MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20)
        librosa.display.specshow(mfcc, sr=sample_rate, hop_length=hop_length,
                                  x_axis='time', ax=axes[0, 1], cmap=cmap)
        axes[0, 1].set_title(tr_("MFCC"), color='#cdd6f4')
        
        # 色度图
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        librosa.display.specshow(chroma, sr=sample_rate, hop_length=hop_length,
                                  x_axis='time', y_axis='chroma', ax=axes[1, 0], cmap=cmap)
        axes[1, 0].set_title(tr_("Chroma"), color='#cdd6f4')
        
        # 频谱对比
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
        frames = range(len(spectral_centroids))
        t = librosa.frames_to_time(frames, sr=sample_rate, hop_length=hop_length)
        axes[1, 1].plot(t, spectral_centroids, color='#89b4fa')
        axes[1, 1].set_title(tr_("Spectral centroid"), color='#cdd6f4')
        axes[1, 1].set_xlabel(tr_("Time (s)"), color='#cdd6f4')
        axes[1, 1].set_ylabel('Hz', color='#cdd6f4')
        
        # 应用样式
        for ax in axes.flat:
            ax.set_facecolor('#181825')
            ax.tick_params(colors='#a6adc8')
            for spine in ax.spines.values():
                spine.set_color('#45475a')
        
        self.fig.patch.set_facecolor('#1e1e2e')
        plt.tight_layout()
        
        return self.fig
    
    def _apply_style(self, title: str):
        """应用统一样式"""
        self.ax.set_facecolor('#181825')
        self.fig.patch.set_facecolor('#1e1e2e')
        
        self.ax.set_title(title, color='#cdd6f4', fontsize=12)
        self.ax.tick_params(colors='#a6adc8')
        
        for spine in self.ax.spines.values():
            spine.set_color('#45475a')
        
        # 设置标签颜色
        self.ax.xaxis.label.set_color('#cdd6f4')
        self.ax.yaxis.label.set_color('#cdd6f4')
        
        plt.tight_layout()
    
    def get_canvas(self) -> Optional[FigureCanvasQTAgg]:
        """获取Qt画布"""
        if self.fig is not None:
            return FigureCanvasQTAgg(self.fig)
        return None
    
    def save(self, path: str, dpi: int = 150):
        """保存图像"""
        if self.fig is not None:
            self.fig.savefig(path, dpi=dpi, facecolor='#1e1e2e', edgecolor='none')
    
    def close(self):
        """关闭图像"""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None

