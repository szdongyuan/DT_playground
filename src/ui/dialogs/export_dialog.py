"""
Model Export Dialog
"""

import os

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QFileDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QProgressBar, QPushButton,
    QVBoxLayout
)


class ExportModelDialog(QDialog):
    """模型导出对话框"""
    
    export_requested = pyqtSignal(dict)  # 导出请求
    
    def __init__(self, parent=None, model=None):
        super().__init__(parent)
        self.model = model
        self._init_ui()
    
    def _init_ui(self):
        """初始化界面"""
        self.setWindowTitle("导出模型")
        self.setMinimumSize(450, 350)
        self.setModal(True)
        
        layout = QVBoxLayout(self)
        
        # 导出路径
        path_group = QGroupBox("导出位置")
        path_layout = QFormLayout(path_group)
        
        path_input_layout = QHBoxLayout()
        self.export_path = QLineEdit()
        self.export_path.setPlaceholderText("选择导出目录...")
        path_input_layout.addWidget(self.export_path)
        
        browse_btn = QPushButton("浏览")
        browse_btn.clicked.connect(self._browse_path)
        path_input_layout.addWidget(browse_btn)
        
        path_layout.addRow("目录:", path_input_layout)
        
        self.model_name = QLineEdit()
        self.model_name.setPlaceholderText("model_audio_classifier")
        self.model_name.setText("audio_model")
        path_layout.addRow("模型名称:", self.model_name)
        
        layout.addWidget(path_group)
        
        # 导出格式
        format_group = QGroupBox("导出格式")
        format_layout = QFormLayout(format_group)
        
        self.format_combo = QComboBox()
        self.format_combo.addItems([
            "TensorFlow SavedModel (.pb)",
            "Keras H5 (.h5)",
            "Keras (.keras)",
            "TensorFlow Lite (.tflite)",
            "ONNX (.onnx)"
        ])
        format_layout.addRow("格式:", self.format_combo)
        
        layout.addWidget(format_group)
        
        # 导出选项
        options_group = QGroupBox("导出选项")
        options_layout = QVBoxLayout(options_group)
        
        self.include_optimizer = QCheckBox("包含优化器状态")
        self.include_optimizer.setChecked(False)
        options_layout.addWidget(self.include_optimizer)
        
        self.include_config = QCheckBox("导出模型配置 (JSON)")
        self.include_config.setChecked(True)
        options_layout.addWidget(self.include_config)
        
        self.include_weights_only = QCheckBox("仅导出权重")
        self.include_weights_only.setChecked(False)
        options_layout.addWidget(self.include_weights_only)
        
        layout.addWidget(options_group)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # 状态标签
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #a6adc8;")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        
        # 按钮
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.cancel_btn)
        
        self.export_btn = QPushButton("导出")
        self.export_btn.setStyleSheet("""
            QPushButton {
                background-color: #a6e3a1;
                color: #1e1e2e;
                font-weight: bold;
            }
        """)
        self.export_btn.clicked.connect(self._export_model)
        btn_layout.addWidget(self.export_btn)
        
        layout.addLayout(btn_layout)
        
        self._apply_styles()
    
    def _browse_path(self):
        """浏览导出路径"""
        path = QFileDialog.getExistingDirectory(self, "选择导出目录")
        if path:
            self.export_path.setText(path)
    
    def _export_model(self):
        """执行模型导出"""
        export_path = self.export_path.text().strip()
        model_name = self.model_name.text().strip()
        
        if not export_path:
            QMessageBox.warning(self, "警告", "请选择导出目录")
            return
        
        if not model_name:
            QMessageBox.warning(self, "警告", "请输入模型名称")
            return
        
        if self.model is None:
            QMessageBox.warning(self, "警告", "没有可导出的模型")
            return
        
        # 构建导出配置
        format_text = self.format_combo.currentText()
        if "SavedModel" in format_text:
            ext = ""
            save_format = "tf"
        elif "H5" in format_text:
            ext = ".h5"
            save_format = "h5"
        elif ".keras" in format_text:
            ext = ".keras"
            save_format = "keras"
        elif "TFLite" in format_text:
            ext = ".tflite"
            save_format = "tflite"
        elif "ONNX" in format_text:
            ext = ".onnx"
            save_format = "onnx"
        else:
            ext = ".h5"
            save_format = "h5"
        
        config = {
            'path': os.path.join(export_path, model_name + ext),
            'format': save_format,
            'include_optimizer': self.include_optimizer.isChecked(),
            'include_config': self.include_config.isChecked(),
            'weights_only': self.include_weights_only.isChecked()
        }
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # 无限进度
        self.status_label.setText("正在导出模型...")
        self.export_btn.setEnabled(False)
        
        try:
            self._do_export(config)
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(100)
            self.status_label.setText("导出成功!")
            self.status_label.setStyleSheet("color: #a6e3a1;")
            
            QMessageBox.information(
                self, "成功", 
                f"模型已导出到:\n{config['path']}"
            )
            
            self.export_requested.emit(config)
            self.accept()
            
        except Exception as e:
            self.progress_bar.setVisible(False)
            self.status_label.setText(f"导出失败: {str(e)}")
            self.status_label.setStyleSheet("color: #f38ba8;")
            self.export_btn.setEnabled(True)
            
            QMessageBox.critical(self, "错误", f"导出失败:\n{str(e)}")
    
    def _do_export(self, config: dict):
        """执行实际的导出操作"""
        save_format = config['format']
        path = config['path']
        
        if save_format in ['tf', 'h5', 'keras']:
            self.model.save(
                path,
                include_optimizer=config['include_optimizer']
            )
        
        elif save_format == 'tflite':
            import tensorflow as tf
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            tflite_model = converter.convert()
            with open(path, 'wb') as f:
                f.write(tflite_model)
        
        elif save_format == 'onnx':
            # 需要 tf2onnx
            try:
                import tf2onnx
                import tensorflow as tf
                
                spec = (tf.TensorSpec(self.model.input_shape, tf.float32),)
                model_proto, _ = tf2onnx.convert.from_keras(
                    self.model, input_signature=spec
                )
                with open(path, 'wb') as f:
                    f.write(model_proto.SerializeToString())
            except ImportError:
                raise ImportError("请安装 tf2onnx: pip install tf2onnx")
        
        # 导出配置文件
        if config['include_config']:
            import json
            config_path = path.rsplit('.', 1)[0] + '_config.json'
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.model.get_config(), f, indent=2, ensure_ascii=False)
    
    def _apply_styles(self):
        """应用样式"""
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e2e;
                color: #cdd6f4;
            }
            QLabel {
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
            QLineEdit, QComboBox {
                background-color: #313244;
                border: 1px solid #45475a;
                border-radius: 4px;
                padding: 6px;
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
            QProgressBar {
                border: 1px solid #45475a;
                border-radius: 4px;
                background-color: #313244;
                text-align: center;
                color: #cdd6f4;
            }
            QProgressBar::chunk {
                background-color: #a6e3a1;
                border-radius: 3px;
            }
        """)

