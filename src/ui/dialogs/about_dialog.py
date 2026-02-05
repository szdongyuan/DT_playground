"""
About Dialog
"""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QPixmap
from PyQt6.QtWidgets import (
    QDialog, QFrame, QHBoxLayout, QLabel,
    QPushButton, QVBoxLayout
)

from src.ui.i18n import tr_
class AboutDialog(QDialog):
    """关于对话框"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
    
    def _init_ui(self):
        """初始化界面"""
        self.setWindowTitle(tr_("About"))
        self.setFixedSize(450, 380)
        self.setModal(True)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Logo/图标区域
        icon_label = QLabel("🎵")
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_label.setStyleSheet("font-size: 64px;")
        layout.addWidget(icon_label)
        
        # 应用名称
        name_label = QLabel(tr_("AI Acoustic Signal Training Platform"))
        name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name_font = QFont()
        name_font.setPointSize(18)
        name_font.setBold(True)
        name_label.setFont(name_font)
        name_label.setStyleSheet("color: #89b4fa;")
        layout.addWidget(name_label)
        
        # 版本号
        version_label = QLabel(tr_("Version 1.0.0"))
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        version_label.setStyleSheet("color: #a6adc8;")
        layout.addWidget(version_label)
        
        # 分隔线
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("background-color: #45475a;")
        layout.addWidget(line)
        
        # 描述
        desc_label = QLabel(
            tr_(
                "A local training platform for learning AI acoustic signal processing.\n"
                "Supports audio visualization, feature extraction, model building, and training monitoring."
            )
        )
        desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #cdd6f4;")
        layout.addWidget(desc_label)
        
        # 技术栈
        tech_label = QLabel(
            tr_("Tech stack: Python • TensorFlow • PyQt6 • Librosa")
        )
        tech_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tech_label.setStyleSheet("color: #a6adc8; font-size: 11px;")
        layout.addWidget(tech_label)
        
        layout.addStretch()
        
        # 版权信息
        copyright_label = QLabel("© 2024 AI Audio Training Team")
        copyright_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        copyright_label.setStyleSheet("color: #6c7086; font-size: 11px;")
        layout.addWidget(copyright_label)
        
        # 关闭按钮
        close_btn = QPushButton(tr_("Close"))
        close_btn.setFixedWidth(100)
        close_btn.clicked.connect(self.accept)
        
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        self._apply_styles()
    
    def _apply_styles(self):
        """应用样式"""
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e2e;
                color: #cdd6f4;
            }
            QPushButton {
                background-color: #45475a;
                color: #cdd6f4;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #585b70;
            }
        """)

