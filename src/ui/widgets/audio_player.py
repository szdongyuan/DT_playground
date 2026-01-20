"""
音频播放控件
"""

import os
import tempfile

import numpy as np
import soundfile as sf

from PyQt6.QtCore import Qt, QTimer, QUrl, pyqtSignal
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton, QSlider,
    QStyle, QWidget
)


class AudioPlayerWidget(QWidget):
    """音频播放控制控件"""
    
    # 信号
    position_changed = pyqtSignal(float)  # 播放位置改变（秒）
    playback_state_changed = pyqtSignal(bool)  # 播放状态改变
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.duration = 0
        self._init_ui()
        self._init_player()
    
    def _init_ui(self):
        """初始化界面"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        
        # 播放按钮
        self.play_btn = QPushButton("▶")
        self.play_btn.setFixedSize(40, 40)
        self.play_btn.setStyleSheet("""
            QPushButton {
                background-color: #a6e3a1;
                color: #1e1e2e;
                border-radius: 20px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #94e2d5;
            }
        """)
        self.play_btn.clicked.connect(self._toggle_playback)
        layout.addWidget(self.play_btn)
        
        # 停止按钮
        self.stop_btn = QPushButton("⬛")
        self.stop_btn.setFixedSize(32, 32)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #f38ba8;
                color: #1e1e2e;
                border-radius: 4px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #eba0ac;
            }
        """)
        self.stop_btn.clicked.connect(self._stop_playback)
        layout.addWidget(self.stop_btn)
        
        # 当前时间
        self.current_time_label = QLabel("00:00")
        self.current_time_label.setStyleSheet("color: #cdd6f4; min-width: 45px;")
        layout.addWidget(self.current_time_label)
        
        # 进度条
        self.progress_slider = QSlider(Qt.Orientation.Horizontal)
        self.progress_slider.setRange(0, 1000)
        self.progress_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #313244;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #89b4fa;
                width: 14px;
                height: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
            QSlider::handle:horizontal:hover {
                background: #b4befe;
            }
            QSlider::sub-page:horizontal {
                background: #89b4fa;
                border-radius: 3px;
            }
        """)
        self.progress_slider.sliderPressed.connect(self._on_slider_pressed)
        self.progress_slider.sliderReleased.connect(self._on_slider_released)
        self.progress_slider.sliderMoved.connect(self._on_slider_moved)
        layout.addWidget(self.progress_slider, stretch=1)
        
        # 总时间
        self.total_time_label = QLabel("00:00")
        self.total_time_label.setStyleSheet("color: #cdd6f4; min-width: 45px;")
        layout.addWidget(self.total_time_label)
        
        # 音量控制
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(70)
        self.volume_slider.setFixedWidth(80)
        self.volume_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #313244;
                height: 4px;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #cba6f7;
                width: 12px;
                height: 12px;
                margin: -4px 0;
                border-radius: 6px;
            }
            QSlider::sub-page:horizontal {
                background: #cba6f7;
                border-radius: 2px;
            }
        """)
        self.volume_slider.valueChanged.connect(self._on_volume_changed)
        
        volume_label = QLabel("🔊")
        layout.addWidget(volume_label)
        layout.addWidget(self.volume_slider)
    
    def _init_player(self):
        """初始化媒体播放器"""
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        
        # 设置初始音量
        self.audio_output.setVolume(0.7)
        
        # 连接信号
        self.player.positionChanged.connect(self._on_position_changed)
        self.player.durationChanged.connect(self._on_duration_changed)
        self.player.playbackStateChanged.connect(self._on_playback_state_changed)
        
        # 是否正在拖动滑块
        self._slider_being_dragged = False
    
    def load_audio(self, audio_path: str):
        """加载音频文件"""
        self.player.setSource(QUrl.fromLocalFile(audio_path))
    
    def _toggle_playback(self):
        """切换播放/暂停状态"""
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()
    
    def _stop_playback(self):
        """停止播放"""
        self.player.stop()
    
    def _on_position_changed(self, position: int):
        """播放位置改变"""
        if not self._slider_being_dragged and self.duration > 0:
            self.progress_slider.setValue(int(position / self.duration * 1000))
        
        # 更新时间标签
        self.current_time_label.setText(self._format_time(position))
        
        # 发送位置信号（秒）
        self.position_changed.emit(position / 1000.0)
    
    def _on_duration_changed(self, duration: int):
        """音频时长改变"""
        self.duration = duration
        self.total_time_label.setText(self._format_time(duration))
    
    def _on_playback_state_changed(self, state):
        """播放状态改变"""
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.play_btn.setText("⏸")
            self.playback_state_changed.emit(True)
        else:
            self.play_btn.setText("▶")
            self.playback_state_changed.emit(False)
    
    def _on_slider_pressed(self):
        """滑块被按下"""
        self._slider_being_dragged = True
    
    def _on_slider_released(self):
        """滑块被释放"""
        self._slider_being_dragged = False
        position = int(self.progress_slider.value() / 1000 * self.duration)
        self.player.setPosition(position)
    
    def _on_slider_moved(self, value: int):
        """滑块被拖动"""
        if self.duration > 0:
            position = int(value / 1000 * self.duration)
            self.current_time_label.setText(self._format_time(position))
    
    def _on_volume_changed(self, value: int):
        """音量改变"""
        self.audio_output.setVolume(value / 100.0)
    
    def _format_time(self, ms: int) -> str:
        """格式化时间（毫秒转为 MM:SS）"""
        seconds = ms // 1000
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes:02d}:{seconds:02d}"
    
    def set_audio(self, audio_data: np.ndarray, sample_rate: int):
        """
        设置音频数据用于播放
        
        Args:
            audio_data: 音频波形数据 (numpy array)
            sample_rate: 采样率
        """
        # 停止当前播放
        self.stop()
        
        # 将numpy数组保存为临时wav文件
        try:
            # 创建临时文件
            temp_file = tempfile.NamedTemporaryFile(
                suffix='.wav', delete=False, prefix='preview_'
            )
            temp_path = temp_file.name
            temp_file.close()
            
            # 确保音频数据是正确的形状
            if len(audio_data.shape) == 1:
                # 单声道
                pass
            elif len(audio_data.shape) == 2 and audio_data.shape[0] == 2:
                # 双声道 (2, N) -> (N, 2)
                audio_data = audio_data.T
            
            # 保存为wav文件
            sf.write(temp_path, audio_data, sample_rate)
            
            # 加载音频文件
            self.load_audio(temp_path)
            
            # 记录临时文件路径以便清理
            if hasattr(self, '_temp_file_path') and self._temp_file_path:
                try:
                    os.unlink(self._temp_file_path)
                except:
                    pass
            self._temp_file_path = temp_path
            
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"设置音频数据失败: {e}")
    
    def stop(self):
        """停止播放"""
        self._stop_playback()
    
    def clear(self):
        """清除播放器状态"""
        self.stop()
        self.progress_slider.setValue(0)
        self.current_time_label.setText("00:00")
        self.total_time_label.setText("00:00")
        self.duration = 0
        
        # 清理临时文件
        if hasattr(self, '_temp_file_path') and self._temp_file_path:
            try:
                os.unlink(self._temp_file_path)
            except:
                pass
            self._temp_file_path = None

