"""
Settings Dialog
"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
    QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSpinBox,
    QTabWidget, QVBoxLayout, QWidget
)


class SettingsDialog(QDialog):
    """应用设置对话框"""
    
    settings_changed = pyqtSignal(dict)
    
    def __init__(self, parent=None, current_settings: dict = None):
        super().__init__(parent)
        self.current_settings = current_settings or {}
        self._init_ui()
        self._load_settings()
    
    def _init_ui(self):
        """初始化界面"""
        self.setWindowTitle("设置")
        self.setMinimumSize(500, 450)
        self.setModal(True)
        
        layout = QVBoxLayout(self)
        
        # 标签页
        self.tabs = QTabWidget()
        
        # 音频设置
        self.tabs.addTab(self._create_audio_tab(), "音频")
        
        # 模型设置
        self.tabs.addTab(self._create_model_tab(), "模型")
        
        # 界面设置
        self.tabs.addTab(self._create_ui_tab(), "界面")
        
        # GPU设置
        self.tabs.addTab(self._create_gpu_tab(), "GPU")
        
        layout.addWidget(self.tabs)
        
        # 按钮
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        self.reset_btn = QPushButton("恢复默认")
        self.reset_btn.clicked.connect(self._reset_defaults)
        btn_layout.addWidget(self.reset_btn)
        
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.cancel_btn)
        
        self.save_btn = QPushButton("保存")
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #a6e3a1;
                color: #1e1e2e;
                font-weight: bold;
            }
        """)
        self.save_btn.clicked.connect(self._save_settings)
        btn_layout.addWidget(self.save_btn)
        
        layout.addLayout(btn_layout)
        
        self._apply_styles()
    
    def _create_audio_tab(self) -> QWidget:
        """创建音频设置标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 采样设置
        sample_group = QGroupBox("采样设置")
        sample_layout = QFormLayout(sample_group)
        
        self.sample_rate_combo = QComboBox()
        self.sample_rate_combo.addItems(["22050", "44100", "48000", "96000"])
        self.sample_rate_combo.setCurrentText("44100")
        sample_layout.addRow("采样率 (Hz):", self.sample_rate_combo)
        
        self.mono_check = QCheckBox("转换为单声道")
        self.mono_check.setChecked(True)
        sample_layout.addRow("", self.mono_check)
        
        layout.addWidget(sample_group)
        
        # 频谱分析设置
        spectrum_group = QGroupBox("频谱分析")
        spectrum_layout = QFormLayout(spectrum_group)
        
        self.n_fft_combo = QComboBox()
        self.n_fft_combo.addItems(["512", "1024", "2048", "4096"])
        self.n_fft_combo.setCurrentText("2048")
        spectrum_layout.addRow("FFT窗口大小:", self.n_fft_combo)
        
        self.hop_length_spin = QSpinBox()
        self.hop_length_spin.setRange(128, 2048)
        self.hop_length_spin.setValue(512)
        self.hop_length_spin.setSingleStep(128)
        spectrum_layout.addRow("帧移 (Hop Length):", self.hop_length_spin)
        
        self.n_mels_spin = QSpinBox()
        self.n_mels_spin.setRange(40, 256)
        self.n_mels_spin.setValue(128)
        spectrum_layout.addRow("Mel滤波器数量:", self.n_mels_spin)
        
        self.n_mfcc_spin = QSpinBox()
        self.n_mfcc_spin.setRange(10, 40)
        self.n_mfcc_spin.setValue(20)
        spectrum_layout.addRow("MFCC系数数量:", self.n_mfcc_spin)
        
        layout.addWidget(spectrum_group)
        
        # 预处理设置
        preprocess_group = QGroupBox("预处理")
        preprocess_layout = QFormLayout(preprocess_group)
        
        self.normalize_check = QCheckBox("音频归一化")
        self.normalize_check.setChecked(True)
        preprocess_layout.addRow("", self.normalize_check)
        
        self.remove_silence_check = QCheckBox("移除静音段")
        self.remove_silence_check.setChecked(False)
        preprocess_layout.addRow("", self.remove_silence_check)
        
        self.silence_threshold_spin = QSpinBox()
        self.silence_threshold_spin.setRange(10, 60)
        self.silence_threshold_spin.setValue(20)
        self.silence_threshold_spin.setSuffix(" dB")
        preprocess_layout.addRow("静音阈值:", self.silence_threshold_spin)
        
        layout.addWidget(preprocess_group)
        layout.addStretch()
        
        return widget
    
    def _create_model_tab(self) -> QWidget:
        """创建模型设置标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 默认训练参数
        train_group = QGroupBox("默认训练参数")
        train_layout = QFormLayout(train_group)
        
        self.default_epochs_spin = QSpinBox()
        self.default_epochs_spin.setRange(1, 1000)
        self.default_epochs_spin.setValue(50)
        train_layout.addRow("Epochs:", self.default_epochs_spin)
        
        self.default_batch_spin = QSpinBox()
        self.default_batch_spin.setRange(1, 512)
        self.default_batch_spin.setValue(32)
        train_layout.addRow("Batch Size:", self.default_batch_spin)
        
        self.default_lr_spin = QDoubleSpinBox()
        self.default_lr_spin.setRange(0.00001, 1.0)
        self.default_lr_spin.setDecimals(5)
        self.default_lr_spin.setSingleStep(0.0001)
        self.default_lr_spin.setValue(0.001)
        train_layout.addRow("学习率:", self.default_lr_spin)
        
        layout.addWidget(train_group)
        
        # 模型保存设置
        save_group = QGroupBox("模型保存")
        save_layout = QFormLayout(save_group)
        
        model_path_layout = QHBoxLayout()
        self.model_save_path = QLineEdit()
        self.model_save_path.setPlaceholderText("选择模型保存目录...")
        model_path_layout.addWidget(self.model_save_path)
        
        browse_btn = QPushButton("浏览")
        browse_btn.clicked.connect(self._browse_model_path)
        model_path_layout.addWidget(browse_btn)
        
        save_layout.addRow("保存路径:", model_path_layout)
        
        self.auto_save_check = QCheckBox("自动保存最佳模型")
        self.auto_save_check.setChecked(True)
        save_layout.addRow("", self.auto_save_check)
        
        self.save_format_combo = QComboBox()
        self.save_format_combo.addItems(["SavedModel", "H5", "Keras"])
        save_layout.addRow("保存格式:", self.save_format_combo)
        
        layout.addWidget(save_group)
        
        # 早停设置
        early_stop_group = QGroupBox("早停 (Early Stopping)")
        early_stop_layout = QFormLayout(early_stop_group)
        
        self.early_stop_check = QCheckBox("启用早停")
        self.early_stop_check.setChecked(True)
        early_stop_layout.addRow("", self.early_stop_check)
        
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(1, 100)
        self.patience_spin.setValue(10)
        early_stop_layout.addRow("耐心值 (Patience):", self.patience_spin)
        
        self.min_delta_spin = QDoubleSpinBox()
        self.min_delta_spin.setRange(0.0, 0.1)
        self.min_delta_spin.setDecimals(4)
        self.min_delta_spin.setSingleStep(0.0001)
        self.min_delta_spin.setValue(0.0001)
        early_stop_layout.addRow("最小改进量:", self.min_delta_spin)
        
        layout.addWidget(early_stop_group)
        layout.addStretch()
        
        return widget
    
    def _create_ui_tab(self) -> QWidget:
        """创建界面设置标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 主题设置
        theme_group = QGroupBox("主题")
        theme_layout = QFormLayout(theme_group)
        
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["深色", "浅色", "跟随系统"])
        theme_layout.addRow("主题:", self.theme_combo)
        
        layout.addWidget(theme_group)
        
        # 可视化设置
        viz_group = QGroupBox("可视化")
        viz_layout = QFormLayout(viz_group)
        
        self.waveform_color = QComboBox()
        self.waveform_color.addItems(["蓝色", "绿色", "紫色", "橙色"])
        viz_layout.addRow("波形颜色:", self.waveform_color)
        
        self.spectrogram_cmap = QComboBox()
        self.spectrogram_cmap.addItems(["viridis", "magma", "plasma", "inferno", "jet"])
        viz_layout.addRow("频谱图配色:", self.spectrogram_cmap)
        
        self.show_grid_check = QCheckBox("显示网格")
        self.show_grid_check.setChecked(True)
        viz_layout.addRow("", self.show_grid_check)
        
        layout.addWidget(viz_group)
        
        # 更新频率
        update_group = QGroupBox("更新设置")
        update_layout = QFormLayout(update_group)
        
        self.chart_update_spin = QSpinBox()
        self.chart_update_spin.setRange(100, 5000)
        self.chart_update_spin.setValue(500)
        self.chart_update_spin.setSuffix(" ms")
        update_layout.addRow("图表更新间隔:", self.chart_update_spin)
        
        layout.addWidget(update_group)
        layout.addStretch()
        
        return widget
    
    def _create_gpu_tab(self) -> QWidget:
        """创建GPU设置标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # GPU设置
        gpu_group = QGroupBox("GPU设置")
        gpu_layout = QFormLayout(gpu_group)
        
        self.use_gpu_check = QCheckBox("使用GPU加速")
        self.use_gpu_check.setChecked(True)
        gpu_layout.addRow("", self.use_gpu_check)
        
        self.gpu_memory_spin = QSpinBox()
        self.gpu_memory_spin.setRange(0, 100)
        self.gpu_memory_spin.setValue(0)
        self.gpu_memory_spin.setSuffix(" % (0=自动)")
        gpu_layout.addRow("显存限制:", self.gpu_memory_spin)
        
        self.mixed_precision_check = QCheckBox("混合精度训练")
        self.mixed_precision_check.setChecked(False)
        gpu_layout.addRow("", self.mixed_precision_check)
        
        layout.addWidget(gpu_group)
        
        # GPU信息
        info_group = QGroupBox("GPU信息")
        info_layout = QVBoxLayout(info_group)
        
        self.gpu_info_label = QLabel("正在检测GPU...")
        self.gpu_info_label.setWordWrap(True)
        info_layout.addWidget(self.gpu_info_label)
        
        detect_btn = QPushButton("重新检测")
        detect_btn.clicked.connect(self._detect_gpu)
        info_layout.addWidget(detect_btn)
        
        layout.addWidget(info_group)
        layout.addStretch()
        
        # 初始检测GPU
        self._detect_gpu()
        
        return widget
    
    def _detect_gpu(self):
        """检测GPU"""
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                info_text = f"检测到 {len(gpus)} 个GPU:\n"
                for i, gpu in enumerate(gpus):
                    info_text += f"  {i+1}. {gpu.name}\n"
                self.gpu_info_label.setText(info_text)
                self.gpu_info_label.setStyleSheet("color: #a6e3a1;")
            else:
                self.gpu_info_label.setText("未检测到GPU，将使用CPU进行训练")
                self.gpu_info_label.setStyleSheet("color: #f9e2af;")
        except Exception as e:
            self.gpu_info_label.setText(f"检测GPU时出错: {str(e)}")
            self.gpu_info_label.setStyleSheet("color: #f38ba8;")
    
    def _browse_model_path(self):
        """浏览模型保存路径"""
        path = QFileDialog.getExistingDirectory(self, "选择模型保存目录")
        if path:
            self.model_save_path.setText(path)
    
    def _load_settings(self):
        """加载当前设置"""
        if not self.current_settings:
            return
        
        # 音频设置
        if 'sample_rate' in self.current_settings:
            self.sample_rate_combo.setCurrentText(str(self.current_settings['sample_rate']))
        if 'mono' in self.current_settings:
            self.mono_check.setChecked(self.current_settings['mono'])
        # ... 加载更多设置
    
    def _save_settings(self):
        """保存设置"""
        settings = {
            # 音频设置
            'sample_rate': int(self.sample_rate_combo.currentText()),
            'mono': self.mono_check.isChecked(),
            'n_fft': int(self.n_fft_combo.currentText()),
            'hop_length': self.hop_length_spin.value(),
            'n_mels': self.n_mels_spin.value(),
            'n_mfcc': self.n_mfcc_spin.value(),
            'normalize': self.normalize_check.isChecked(),
            'remove_silence': self.remove_silence_check.isChecked(),
            'silence_threshold': self.silence_threshold_spin.value(),
            
            # 模型设置
            'default_epochs': self.default_epochs_spin.value(),
            'default_batch_size': self.default_batch_spin.value(),
            'default_learning_rate': self.default_lr_spin.value(),
            'model_save_path': self.model_save_path.text(),
            'auto_save_best': self.auto_save_check.isChecked(),
            'save_format': self.save_format_combo.currentText(),
            'early_stopping': self.early_stop_check.isChecked(),
            'patience': self.patience_spin.value(),
            'min_delta': self.min_delta_spin.value(),
            
            # 界面设置
            'theme': self.theme_combo.currentText(),
            'waveform_color': self.waveform_color.currentText(),
            'spectrogram_cmap': self.spectrogram_cmap.currentText(),
            'show_grid': self.show_grid_check.isChecked(),
            'chart_update_interval': self.chart_update_spin.value(),
            
            # GPU设置
            'use_gpu': self.use_gpu_check.isChecked(),
            'gpu_memory_limit': self.gpu_memory_spin.value(),
            'mixed_precision': self.mixed_precision_check.isChecked()
        }
        
        self.settings_changed.emit(settings)
        self.accept()
    
    def _reset_defaults(self):
        """恢复默认设置"""
        # 音频
        self.sample_rate_combo.setCurrentText("44100")
        self.mono_check.setChecked(True)
        self.n_fft_combo.setCurrentText("2048")
        self.hop_length_spin.setValue(512)
        self.n_mels_spin.setValue(128)
        self.n_mfcc_spin.setValue(20)
        self.normalize_check.setChecked(True)
        self.remove_silence_check.setChecked(False)
        self.silence_threshold_spin.setValue(20)
        
        # 模型
        self.default_epochs_spin.setValue(50)
        self.default_batch_spin.setValue(32)
        self.default_lr_spin.setValue(0.001)
        self.auto_save_check.setChecked(True)
        self.save_format_combo.setCurrentIndex(0)
        self.early_stop_check.setChecked(True)
        self.patience_spin.setValue(10)
        self.min_delta_spin.setValue(0.0001)
        
        # 界面
        self.theme_combo.setCurrentIndex(0)
        self.waveform_color.setCurrentIndex(0)
        self.spectrogram_cmap.setCurrentIndex(0)
        self.show_grid_check.setChecked(True)
        self.chart_update_spin.setValue(500)
        
        # GPU
        self.use_gpu_check.setChecked(True)
        self.gpu_memory_spin.setValue(0)
        self.mixed_precision_check.setChecked(False)
    
    def _apply_styles(self):
        """应用样式"""
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e2e;
                color: #cdd6f4;
            }
            QWidget {
                color: #cdd6f4;
            }
            QLabel {
                color: #cdd6f4;
            }
            QTabWidget::pane {
                border: 1px solid #45475a;
                background-color: #181825;
            }
            QTabBar::tab {
                background-color: #313244;
                color: #bac2de;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #45475a;
                color: #cdd6f4;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #45475a;
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 10px;
                color: #cdd6f4;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #89b4fa;
            }
            QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {
                background-color: #313244;
                border: 1px solid #45475a;
                border-radius: 4px;
                padding: 4px;
                min-width: 120px;
                color: #cdd6f4;
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
            QComboBox QAbstractItemView {
                background-color: #313244;
                color: #cdd6f4;
                selection-background-color: #45475a;
            }
            QCheckBox {
                color: #cdd6f4;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #45475a;
                border-radius: 3px;
                background-color: #313244;
            }
            QCheckBox::indicator:checked {
                background-color: #89b4fa;
                border-color: #89b4fa;
            }
            QPushButton {
                background-color: #45475a;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                color: #cdd6f4;
            }
            QPushButton:hover {
                background-color: #585b70;
            }
        """)

