"""
波形显示控件
"""

import numpy as np

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

try:
    import pyqtgraph as pg
    from PyQt6.QtGui import QFont
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False


class WaveformWidget(QWidget):
    """波形可视化控件"""
    
    # 信号
    region_selected = pyqtSignal(float, float)  # 选区起止时间
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.audio_data = None
        self.sample_rate = 44100
        self.playback_position = 0.0
        self._init_ui()
    
    def _init_ui(self):
        """初始化界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 标题
        title = QLabel("波形显示")
        title.setStyleSheet("font-weight: bold; color: #89b4fa;")
        layout.addWidget(title)
        
        if HAS_PYQTGRAPH:
            # 使用PyQtGraph绑定波形
            self.plot_widget = pg.PlotWidget()
            self.plot_widget.setBackground('#181825')
            self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
            
            # 设置字体以避免字体大小问题
            font = QFont("Microsoft YaHei", 9)
            
            # 设置坐标轴
            for axis_name in ['left', 'bottom']:
                axis = self.plot_widget.getPlotItem().getAxis(axis_name)
                axis.setTickFont(font)
                axis.setStyle(tickTextOffset=5)
            
            self.plot_widget.setLabel('left', text='振幅')
            self.plot_widget.setLabel('bottom', text='时间 (s)')
            
            # 波形曲线
            self.waveform_curve = self.plot_widget.plot(
                pen=pg.mkPen(color='#89b4fa', width=1)
            )
            
            # 播放位置线
            self.playback_line = pg.InfiniteLine(
                pos=0, angle=90, 
                pen=pg.mkPen(color='#f38ba8', width=2)
            )
            self.plot_widget.addItem(self.playback_line)
            
            # 设置最小高度，与频谱图保持一致
            self.plot_widget.setMinimumHeight(200)
            
            layout.addWidget(self.plot_widget)
        else:
            # 后备方案：简单标签
            self.placeholder = QLabel("请安装 pyqtgraph 以显示波形")
            self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.placeholder.setStyleSheet("""
                background-color: #181825;
                border: 1px dashed #45475a;
                border-radius: 4px;
                padding: 40px;
            """)
            layout.addWidget(self.placeholder)
    
    def load_audio(self, audio_path: str):
        """加载音频文件并显示波形"""
        try:
            import librosa
            import warnings
            
            # 加载音频，抑制警告
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.audio_data, self.sample_rate = librosa.load(
                    audio_path, sr=self.sample_rate, mono=True
                )
            
            # 验证音频数据
            if self.audio_data is not None and len(self.audio_data) > 0:
                self._update_waveform()
            
        except Exception as e:
            print(f"加载音频失败: {e}")
    
    def _update_waveform(self):
        """更新波形显示"""
        if self.audio_data is None or not HAS_PYQTGRAPH:
            return
        
        audio = self.audio_data
        
        # 确保数据是 1D 数组
        if len(audio.shape) > 1:
            # 多维数组，取平均（如立体声取单声道）
            audio = np.mean(audio, axis=0)
            # 如果仍然是多维，继续处理
            while len(audio.shape) > 1:
                audio = np.mean(audio, axis=0)
        
        # 降采样以提高性能
        max_points = 10000
        if len(audio) > max_points:
            step = len(audio) // max_points
            display_data = audio[::step]
        else:
            display_data = audio
        
        # 计算时间轴
        duration = len(audio) / self.sample_rate
        time_axis = np.linspace(0, duration, len(display_data))
        
        self.waveform_curve.setData(time_axis, display_data)
        self.plot_widget.setXRange(0, duration)
        self.plot_widget.setYRange(-1, 1)
    
    def set_playback_position(self, position: float):
        """设置播放位置指示器"""
        self.playback_position = position
        if HAS_PYQTGRAPH:
            self.playback_line.setPos(position)
    
    def set_audio_data(self, audio_data: np.ndarray, sample_rate: int):
        """直接设置音频数据"""
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self._update_waveform()
    
    def set_audio(self, audio_data: np.ndarray, sample_rate: int):
        """设置音频数据（别名）"""
        self.set_audio_data(audio_data, sample_rate)
    
    def set_curve(self, data: np.ndarray, name: str = "数据"):
        """显示任意曲线数据"""
        if not HAS_PYQTGRAPH:
            return
        
        # 确保数据是 1D 数组
        if len(data.shape) == 0:
            # 标量，转为数组
            data = np.array([data])
        elif len(data.shape) > 1:
            # 多维数组，取时间维度的均值（展平为 1D）
            data = np.mean(data, axis=0)
            # 如果仍然是多维，继续展平
            while len(data.shape) > 1:
                data = np.mean(data, axis=0)
        
        x = np.arange(len(data))
        self.waveform_curve.setData(x, data)
        self.plot_widget.setXRange(0, len(data))
        
        # 自动调整Y轴范围
        if len(data) > 0:
            y_min, y_max = np.min(data), np.max(data)
            margin = (y_max - y_min) * 0.1 if y_max != y_min else 1
            self.plot_widget.setYRange(y_min - margin, y_max + margin)
        
        self.plot_widget.setLabel('bottom', text='样本')
        self.plot_widget.setLabel('left', text=name)
    
    def clear(self):
        """清除显示"""
        if HAS_PYQTGRAPH:
            self.waveform_curve.clear()
            self.audio_data = None
