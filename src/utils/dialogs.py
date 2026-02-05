"""
Common Dialog Utilities
Provides dialogs with copyable error text and automatic logging
"""

import logging
import traceback

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication, QDialog, QHBoxLayout, QLabel,
    QPushButton, QTextEdit, QVBoxLayout
)

from src.ui.i18n import tr_
from src.ui.styles import Styles

logger = logging.getLogger('AudioTrainingApp')


class ErrorDialog(QDialog):
    """可复制文本的错误对话框"""
    
    def __init__(self, title: str, message: str, details: str = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(500)
        self.setMinimumHeight(200)
        
        # 记录到日志
        log_msg = f"{title}: {message}"
        if details:
            log_msg += f"\n{details}"
        logger.error(log_msg)
        
        self._init_ui(message, details)
    
    def _init_ui(self, message: str, details: str):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # 样式
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {Styles.COLORS['base']};
            }}
            QLabel {{
                color: {Styles.COLORS['text']};
            }}
            QPushButton {{
                background-color: {Styles.COLORS['surface0']};
                color: {Styles.COLORS['text']};
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }}
            QPushButton:hover {{
                background-color: {Styles.COLORS['surface1']};
            }}
            QTextEdit {{
                background-color: {Styles.COLORS['mantle']};
                color: {Styles.COLORS['text']};
                border: 1px solid {Styles.COLORS['surface1']};
                border-radius: 4px;
                font-family: Consolas, monospace;
                font-size: 11px;
            }}
        """)
        
        # 图标和标题
        header = QHBoxLayout()
        icon_label = QLabel("❌")
        icon_label.setStyleSheet(f"font-size: 32px; color: {Styles.COLORS['red']};")
        header.addWidget(icon_label)
        
        title_label = QLabel(tr_("An error occurred"))
        title_label.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {Styles.COLORS['red']};")
        header.addWidget(title_label)
        header.addStretch()
        layout.addLayout(header)
        
        # 错误消息
        msg_label = QLabel(message)
        msg_label.setWordWrap(True)
        msg_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        msg_label.setStyleSheet("font-size: 13px; margin: 10px 0;")
        layout.addWidget(msg_label)
        
        # 详细信息（可选）
        if details:
            details_label = QLabel(tr_("Details:"))
            details_label.setStyleSheet(f"color: {Styles.COLORS['subtext']}; font-size: 11px;")
            layout.addWidget(details_label)
            
            self.details_text = QTextEdit()
            self.details_text.setPlainText(details)
            self.details_text.setReadOnly(True)
            self.details_text.setMinimumHeight(120)
            layout.addWidget(self.details_text)
        
        # 按钮
        btn_layout = QHBoxLayout()
        
        copy_btn = QPushButton(tr_("📋 Copy error details"))
        copy_btn.clicked.connect(lambda: self._copy_to_clipboard(message, details))
        btn_layout.addWidget(copy_btn)
        
        btn_layout.addStretch()
        
        close_btn = QPushButton(tr_("Close"))
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Styles.COLORS['red']};
                color: {Styles.COLORS['base']};
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #eba0ac;
            }}
        """)
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        
        layout.addLayout(btn_layout)
    
    def _copy_to_clipboard(self, message: str, details: str):
        """复制到剪贴板"""
        text = message
        if details:
            text += "\n\n" + tr_("Details:") + "\n" + details
        
        clipboard = QApplication.clipboard()
        clipboard.setText(text)


def show_error(parent, title: str, message: str, exception: Exception = None):
    """
    显示错误对话框，自动记录日志
    
    Args:
        parent: 父窗口
        title: 对话框标题
        message: 错误消息
        exception: 异常对象（可选，用于获取堆栈跟踪）
    """
    details = None
    if exception:
        details = ''.join(traceback.format_exception(type(exception), exception, exception.__traceback__))
    
    dialog = ErrorDialog(title, message, details, parent)
    dialog.exec()


def show_warning(parent, title: str, message: str):
    """显示警告对话框，记录日志"""
    logger.warning(f"{title}: {message}")
    
    from PyQt6.QtWidgets import QMessageBox
    QMessageBox.warning(parent, title, message)


def show_info(parent, title: str, message: str):
    """显示信息对话框，记录日志"""
    logger.info(f"{title}: {message}")
    
    from PyQt6.QtWidgets import QMessageBox
    QMessageBox.information(parent, title, message)

