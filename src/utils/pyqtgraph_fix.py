"""
PyQtGraph Compatibility Fix
Resolves font compatibility issues between PyQtGraph and PyQt6
"""

import sys
import os


def suppress_qt_font_warnings():
    """
    抑制Qt字体相关警告
    设置Qt环境变量来抑制警告
    """
    # 设置Qt环境变量来抑制某些警告
    os.environ['QT_LOGGING_RULES'] = '*.debug=false;qt.qpa.*=false'


def install_qt_message_handler():
    """
    安装Qt消息处理器来过滤字体警告
    必须在QApplication创建后调用
    """
    try:
        from PyQt6.QtCore import qInstallMessageHandler, QtMsgType
        
        def message_handler(msg_type, context, message):
            # 过滤字体警告
            if message:
                if 'QFont::setPointSize' in message:
                    return  # 忽略这个消息
                if 'Point size <= 0' in message:
                    return  # 忽略这个消息
            
            # 其他消息正常输出到原始stderr
            try:
                if msg_type == QtMsgType.QtWarningMsg:
                    sys.__stderr__.write(f"Warning: {message}\n")
                elif msg_type == QtMsgType.QtCriticalMsg:
                    sys.__stderr__.write(f"Critical: {message}\n")
                elif msg_type == QtMsgType.QtFatalMsg:
                    sys.__stderr__.write(f"Fatal: {message}\n")
                sys.__stderr__.flush()
            except Exception:
                pass  # 忽略输出错误
        
        qInstallMessageHandler(message_handler)
        return True
    except Exception as e:
        try:
            sys.__stderr__.write(f"无法安装Qt消息处理器: {e}\n")
        except Exception:
            pass
        return False
