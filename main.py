"""
AI声学信号训练平台 - 主入口
用于声学信号的AI处理学习
"""

import logging
import os
import sys
import traceback
import warnings
from datetime import datetime
import argparse
import json


def get_app_root():
    """获取应用程序根目录（兼容 PyInstaller 打包）"""
    if getattr(sys, 'frozen', False):
        # PyInstaller 打包后的运行环境
        return os.path.dirname(sys.executable)
    else:
        # 开发模式
        return os.path.dirname(os.path.abspath(__file__))


# 设置日志
app_root = get_app_root()
log_dir = os.path.join(app_root, 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'app_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

# 配置根日志器
logging.basicConfig(
    level=logging.INFO,  # 改为INFO级别，过滤DEBUG
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.__stdout__)
    ]
)

# 抑制第三方库的日志噪音
for noisy_logger in [
    'matplotlib', 'matplotlib.font_manager',
    'numba', 'numba.core', 'numba.core.byteflow', 'numba.core.interpreter',
    'tensorflow', 'h5py', 'h5py._conv',
    'PIL', 'urllib3', 'asyncio',
]:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

logger = logging.getLogger('AudioTrainingApp')
logger.info(f"日志文件: {log_file}")

# 增加递归限制
sys.setrecursionlimit(10000)

# 抑制一些常见警告
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 抑制TensorFlow日志

# Apply the lightest possible Qt logging suppression before splash appears.
os.environ['QT_LOGGING_RULES'] = '*.debug=false;qt.qpa.*=false'
# Force PyQtGraph to use the same Qt binding as the application.
os.environ['PYQTGRAPH_QT_LIB'] = 'PySide6'

# 必须先导入Qt并创建QApplication
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QIcon


def exception_hook(exc_type, exc_value, exc_tb):    
    """全局异常处理器"""
    logger.error("未捕获的异常:")
    logger.error(''.join(traceback.format_exception(exc_type, exc_value, exc_tb)))
    # 写入日志文件
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write('\n=== 未捕获的异常 ===\n')
        f.write(''.join(traceback.format_exception(exc_type, exc_value, exc_tb)))
    sys.__excepthook__(exc_type, exc_value, exc_tb)


def main():
    """应用程序主入口"""
    logger.info("应用程序启动")
    splash = None

    # Parse restart handshake args (do not interfere with Qt args)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--restart-token", default=None)
    parser.add_argument("--restart-ready-file", default=None)
    restart_args, _unknown = parser.parse_known_args(sys.argv[1:])
    
    # 设置全局异常处理器
    sys.excepthook = exception_hook
    
    # 启用高DPI缩放
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    # 创建QApplication
    app = QApplication(sys.argv)

    from src.ui.startup_splash import StartupSplash, get_app_icon_path

    splash = StartupSplash()
    icon_path = get_app_icon_path()
    if icon_path.exists():
        app.setWindowIcon(QIcon(os.fspath(icon_path)))
        splash.setWindowIcon(QIcon(os.fspath(icon_path)))
    splash.show()
    splash.set_progress(10, "Starting application...")

    try:
        splash.set_progress(18, "Loading AI runtime...")
        try:
            import tensorflow as tf
            tf.keras.config.enable_unsafe_deserialization()
        except Exception:
            pass  # Keras configuration should not block startup.

        # Install i18n after QApplication is created, and before importing UI modules.
        try:
            splash.set_progress(28, "Loading configuration...")
            from src.utils.config import config
            from src.ui.i18n import install as install_i18n

            install_i18n(config.get("ui.language", "zh_CN"))
        except Exception:
            # Never break app startup if i18n init fails.
            pass

        from src.ui.i18n import tr_
        
        # 安装Qt消息处理器来过滤字体警告
        from src.utils.pyqtgraph_fix import install_qt_message_handler
        install_qt_message_handler()
        
        app.setApplicationName(tr_("AI Acoustic Signal Training Platform"))
        app.setApplicationVersion("1.0.0")
        
        # 设置默认字体
        font = QFont("Microsoft YaHei", 10)
        font.setStyleHint(QFont.StyleHint.SansSerif)
        app.setFont(font)
        
        # 配置PyQtGraph（在QApplication创建后）
        try:
            splash.set_progress(36, "Configuring visualization tools...")
            import pyqtgraph as pg
            pg.setConfigOptions(
                antialias=True,
                useOpenGL=False,
                enableExperimental=False
            )
            logger.info("PyQtGraph 配置完成")
        except ImportError:
            logger.warning("PyQtGraph 未安装")
        
        # 在QApplication创建后再导入主窗口模块
        logger.info("加载主窗口模块...")
        splash.set_progress(50, "Loading application modules...")
        from src.app import AudioTrainingApp
        
        # 创建并显示主窗口
        logger.info("创建主窗口...")
        splash.set_progress(60, "Creating main window...")
        window = AudioTrainingApp(startup_progress=splash.set_progress)
        splash.set_progress(100, "Launching application...")
        window.show()
        app.processEvents()
        splash.finish(window)
    except Exception:
        splash.close()
        app.processEvents()
        raise

    # If we were started by a restart request, signal readiness after the window is shown.
    if restart_args.restart_token and restart_args.restart_ready_file:
        def _write_ready_file():
            try:
                ready_path = restart_args.restart_ready_file
                os.makedirs(os.path.dirname(ready_path), exist_ok=True)
                payload = {
                    "token": restart_args.restart_token,
                    "pid": os.getpid(),
                    "ts": datetime.now().isoformat(),
                    "argv": sys.argv,
                }
                with open(ready_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False)
            except Exception:
                # Do not break app startup if handshake fails.
                pass

        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, _write_ready_file)
    
    logger.info("进入主事件循环")
    try:
        sys.exit(app.exec())
    except Exception:
        if splash is not None:
            splash.close()
            app.processEvents()
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("主程序异常:")
        raise
