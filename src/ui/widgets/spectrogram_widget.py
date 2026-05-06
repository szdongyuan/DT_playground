"""
Spectrogram Display Widget
"""

import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QStandardItemModel
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QPushButton,
    QVBoxLayout, QWidget
)

try:
    import pyqtgraph as pg
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False

from ..i18n import tr_
class SpectrogramWidget(QWidget):
    """频谱图可视化控件"""
    
    # 信号
    feature_type_changed = Signal(str)  # 特征类型改变
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.audio_data = None
        self.sample_rate = 44100
        self.spectrogram_data = None
        self._init_ui()
    
    def _init_ui(self):
        """初始化界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 工具栏
        toolbar = QHBoxLayout()
        
        title = QLabel(tr_("Spectrogram"))
        title.setStyleSheet("font-weight: bold; color: #a6e3a1;")
        toolbar.addWidget(title)
        
        toolbar.addStretch()
        
        # 特征类型选择
        self.feature_combo = QComboBox()
        # 按领域分组（用不可选的标题项模拟二级结构）
        self._feature_items = [
            (tr_("— Time-Frequency —"), False),
            (tr_("Mel spectrogram"), True),
            ("MFCC", True),
            ("STFT", True),
            ("CQT", True),
            (tr_("Chroma"), True),
        ]
        for text, _selectable in self._feature_items:
            self.feature_combo.addItem(text)

        # 禁用分组标题项
        model = self.feature_combo.model()
        if isinstance(model, QStandardItemModel):
            for idx, (_text, selectable) in enumerate(self._feature_items):
                if not selectable:
                    item = model.item(idx)
                    if item:
                        item.setEnabled(False)

        self.feature_combo.setStyleSheet("""
            QComboBox {
                background-color: #313244;
                border: 1px solid #45475a;
                border-radius: 4px;
                padding: 4px 8px;
                min-width: 100px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid #cdd6f4;
            }
        """)
        self.feature_combo.currentTextChanged.connect(self._on_feature_changed)
        toolbar.addWidget(self.feature_combo)
        
        layout.addLayout(toolbar)
        
        if HAS_PYQTGRAPH:
            # 图像视图
            self.image_view = pg.ImageView()
            self.image_view.ui.histogram.hide()
            self.image_view.ui.roiBtn.hide()
            self.image_view.ui.menuBtn.hide()
            self.image_view.view.setBackgroundColor('#181825')
            
            # 设置 y 轴方向：低频在下，高频在上
            self.image_view.view.invertY(False)
            
            # 关闭纵横比锁定，使四种频谱图拉伸填满整个视图区域
            # 这样 Mel(128行)、MFCC(20行)、STFT(1025行)、色度图(12行) 显示尺寸一致
            self.image_view.view.setAspectLocked(False)
            
            # 设置颜色映射
            colormap = pg.colormap.get('viridis')
            self.image_view.setColorMap(colormap)
            
            # 设置最小高度，保证频谱图显示一致
            self.image_view.setMinimumHeight(200)
            
            layout.addWidget(self.image_view)
        else:
            self.placeholder = QLabel(tr_("Please install pyqtgraph to display spectrograms"))
            self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.placeholder.setStyleSheet("""
                background-color: #181825;
                border: 1px dashed #45475a;
                border-radius: 4px;
                padding: 40px;
            """)
            layout.addWidget(self.placeholder)
    
    def load_audio(self, audio_path: str):
        """加载音频文件并计算频谱图"""
        try:
            import librosa
            import warnings
            
            # 抑制librosa警告
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.audio_data, self.sample_rate = librosa.load(
                    audio_path, sr=self.sample_rate, mono=True
                )
            
            # 验证音频数据有效
            if self.audio_data is not None and len(self.audio_data) > 2048:
                self._compute_spectrogram()
            
        except Exception as e:
            print(f"Failed to load audio: {e}")
    
    def _compute_spectrogram(self):
        """计算并显示频谱图"""
        if self.audio_data is None or not HAS_PYQTGRAPH:
            return
        
        # 验证音频长度足够
        if len(self.audio_data) < 2048:
            return
        
        try:
            import librosa
            import warnings
            
            feature_type = self.feature_combo.currentText()

            # 分组标题项：不计算
            if feature_type.startswith("—") and feature_type.endswith("—"):
                return
            
            # 抑制librosa警告
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                if feature_type == tr_("Mel spectrogram"):
                    spec = librosa.feature.melspectrogram(
                        y=self.audio_data, sr=self.sample_rate,
                        n_mels=128, fmax=8000
                    )
                    spec_db = librosa.power_to_db(spec, ref=np.max)
                    
                elif feature_type == "MFCC":
                    spec_db = librosa.feature.mfcc(
                        y=self.audio_data, sr=self.sample_rate, n_mfcc=20
                    )
                    
                elif feature_type == "STFT":
                    spec = np.abs(librosa.stft(self.audio_data))
                    spec_db = librosa.amplitude_to_db(spec, ref=np.max)
                    
                elif feature_type == "CQT":
                    spec = np.abs(librosa.cqt(
                        y=self.audio_data,
                        sr=self.sample_rate,
                        hop_length=512,
                        n_bins=84,
                        bins_per_octave=12,
                        fmin=32.7
                    ))
                    spec_db = librosa.amplitude_to_db(spec, ref=np.max)
                    
                elif feature_type == tr_("Chroma"):
                    spec_db = librosa.feature.chroma_stft(
                        y=self.audio_data, sr=self.sample_rate
                    )
                else:
                    return
            
            self.spectrogram_data = spec_db
            # 转置使时间为x轴，频率为y轴
            # 频谱图数据: 行0=低频, 行n=高频
            # 转置后: 列0=低频, 列n=高频 (y轴从下到上)
            self.image_view.setImage(spec_db.T, autoRange=True, autoLevels=True)
            
        except Exception as e:
            print(f"Failed to compute spectrogram: {e}")
    
    def _on_feature_changed(self, feature_type: str):
        """特征类型改变时重新计算"""
        self._compute_spectrogram()
        self.feature_type_changed.emit(feature_type)
    
    def set_audio_data(self, audio_data: np.ndarray, sample_rate: int):
        """直接设置音频数据"""
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self._compute_spectrogram()
    
    def set_audio(self, audio_data: np.ndarray, sample_rate: int):
        """设置音频数据（别名）"""
        self.set_audio_data(audio_data, sample_rate)
    
    def set_feature(self, feature_data: np.ndarray, sample_rate: int, hop_length: int = 512):
        """直接显示特征数据"""
        if not HAS_PYQTGRAPH:
            return
        
        self.spectrogram_data = feature_data
        # 转置使时间为x轴，频率为y轴
        self.image_view.setImage(feature_data.T, autoRange=True, autoLevels=True)
    
    def set_array(self, array: np.ndarray):
        """显示任意2D数组"""
        if not HAS_PYQTGRAPH:
            return
        
        self.spectrogram_data = array
        self.image_view.setImage(array.T, autoRange=True, autoLevels=True)
    
    def clear(self):
        """清除显示"""
        if HAS_PYQTGRAPH:
            self.image_view.clear()
            self.spectrogram_data = None
            self.audio_data = None
