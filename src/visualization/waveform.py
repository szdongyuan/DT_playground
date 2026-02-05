"""
Waveform Visualization Tools
"""

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from src.ui.i18n import tr_
class WaveformPlotter:
    """波形绘制器"""
    
    def __init__(self, figsize: Tuple[int, int] = (10, 3), dpi: int = 100):
        """
        初始化波形绘制器
        
        Args:
            figsize: 图像尺寸
            dpi: 分辨率
        """
        self.figsize = figsize
        self.dpi = dpi
        self.fig = None
        self.ax = None
    
    def plot(self, audio: np.ndarray, 
             sample_rate: int,
             title: str = "Waveform",
             color: str = '#89b4fa',
             show_time: bool = True) -> Figure:
        """
        绘制波形图
        
        Args:
            audio: 音频数据
            sample_rate: 采样率
            title: 标题
            color: 波形颜色
            show_time: 是否显示时间轴
            
        Returns:
            matplotlib Figure对象
        """
        self.fig, self.ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # 计算时间轴
        duration = len(audio) / sample_rate
        time = np.linspace(0, duration, len(audio))
        
        # 绘制波形
        self.ax.plot(time, audio, color=color, linewidth=0.5)
        
        # 设置样式
        self.ax.set_facecolor('#181825')
        self.fig.patch.set_facecolor('#1e1e2e')
        
        self.ax.set_title(title, color='#cdd6f4', fontsize=12)
        self.ax.set_ylabel(tr_("Amplitude"), color='#cdd6f4')
        
        if show_time:
            self.ax.set_xlabel(tr_("Time (s)"), color='#cdd6f4')
        
        self.ax.set_xlim(0, duration)
        self.ax.set_ylim(-1.1, 1.1)
        
        # 网格
        self.ax.grid(True, alpha=0.3, color='#45475a')
        
        # 刻度颜色
        self.ax.tick_params(colors='#a6adc8')
        for spine in self.ax.spines.values():
            spine.set_color('#45475a')
        
        plt.tight_layout()
        return self.fig
    
    def plot_stereo(self, audio: np.ndarray,
                    sample_rate: int,
                    title: str = "Stereo waveform") -> Figure:
        """
        绘制立体声波形
        
        Args:
            audio: 双声道音频数据 (2, n)
            sample_rate: 采样率
            title: 标题
            
        Returns:
            matplotlib Figure对象
        """
        if audio.ndim == 1:
            return self.plot(audio, sample_rate, title)
        
        self.fig, axes = plt.subplots(2, 1, figsize=self.figsize, dpi=self.dpi)
        
        duration = audio.shape[1] / sample_rate
        time = np.linspace(0, duration, audio.shape[1])
        
        channels = [tr_("Left"), tr_("Right")]
        colors = ['#89b4fa', '#a6e3a1']
        
        for i, (ax, channel, color) in enumerate(zip(axes, channels, colors)):
            ax.plot(time, audio[i], color=color, linewidth=0.5)
            ax.set_facecolor('#181825')
            ax.set_ylabel(channel, color='#cdd6f4')
            ax.set_xlim(0, duration)
            ax.set_ylim(-1.1, 1.1)
            ax.grid(True, alpha=0.3, color='#45475a')
            ax.tick_params(colors='#a6adc8')
        
        axes[-1].set_xlabel(tr_("Time (s)"), color='#cdd6f4')
        
        self.fig.patch.set_facecolor('#1e1e2e')
        self.fig.suptitle(title, color='#cdd6f4', fontsize=12)
        plt.tight_layout()
        
        return self.fig
    
    def plot_envelope(self, audio: np.ndarray,
                      sample_rate: int,
                      frame_length: int = 2048,
                      hop_length: int = 512) -> Figure:
        """
        绘制包络线
        
        Args:
            audio: 音频数据
            sample_rate: 采样率
            frame_length: 帧长度
            hop_length: 帧移
            
        Returns:
            matplotlib Figure对象
        """
        import librosa
        
        self.fig, self.ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # 计算RMS能量包络
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        duration = len(audio) / sample_rate
        time_audio = np.linspace(0, duration, len(audio))
        time_rms = np.linspace(0, duration, len(rms))
        
        # 绘制波形
        self.ax.plot(time_audio, audio, color='#45475a', linewidth=0.3, alpha=0.5)
        
        # 绘制包络
        self.ax.plot(
            time_rms,
            rms,
            color='#f38ba8',
            linewidth=2,
            label=tr_("RMS energy"),
        )
        self.ax.plot(time_rms, -rms, color='#f38ba8', linewidth=2)
        
        self.ax.set_facecolor('#181825')
        self.fig.patch.set_facecolor('#1e1e2e')
        
        self.ax.set_title(tr_("Waveform envelope"), color='#cdd6f4')
        self.ax.set_xlabel(tr_("Time (s)"), color='#cdd6f4')
        self.ax.set_ylabel(tr_("Amplitude"), color='#cdd6f4')
        self.ax.legend(facecolor='#313244', edgecolor='#45475a', labelcolor='#cdd6f4')
        
        self.ax.grid(True, alpha=0.3, color='#45475a')
        self.ax.tick_params(colors='#a6adc8')
        
        plt.tight_layout()
        return self.fig
    
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

