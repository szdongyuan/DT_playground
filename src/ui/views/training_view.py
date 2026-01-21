# -*- coding: utf-8 -*-
"""
Training View

Displays training progress, metrics, and results.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtWidgets import (
    QFrame, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QProgressBar, QPushButton, QSplitter, QTableWidget,
    QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget
)

try:
    import pyqtgraph as pg
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False

from ..styles import Styles


logger = logging.getLogger(__name__)


class TrainingView(QWidget):
    """
    训练视图
    
    显示训练进度、损失曲线、指标等信息。
    
    Signals:
        pause_requested: 请求暂停训练
        resume_requested: 请求恢复训练
        stop_requested: 请求停止训练
    """
    
    pause_requested = pyqtSignal()
    resume_requested = pyqtSignal()
    stop_requested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._is_training = False
        self._is_paused = False
        
        # 训练历史数据
        self._epochs: List[int] = []
        self._train_loss: List[float] = []
        self._val_loss: List[float] = []
        self._train_acc: List[float] = []
        self._val_acc: List[float] = []
        
        self._setup_ui()
    
    def _setup_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # 标题和控制栏
        header = self._create_header()
        layout.addWidget(header)
        
        # 主分割器
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)
        layout.addWidget(splitter)
        
        # 左侧：图表
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # 损失曲线
        loss_group = QGroupBox("损失曲线")
        loss_group.setStyleSheet(Styles.group_box(Styles.COLORS['red']))
        loss_layout = QVBoxLayout(loss_group)
        
        if HAS_PYQTGRAPH:
            self._loss_plot = pg.PlotWidget()
            self._loss_plot.setBackground(Styles.COLORS['base'])
            self._loss_plot.showGrid(x=True, y=True, alpha=0.3)
            self._loss_plot.setLabel('left', 'Loss')
            self._loss_plot.setLabel('bottom', 'Epoch')
            loss_layout.addWidget(self._loss_plot)
        else:
            loss_layout.addWidget(QLabel("需要 pyqtgraph 显示图表"))
        
        left_layout.addWidget(loss_group)
        
        # 准确率曲线
        acc_group = QGroupBox("准确率曲线")
        acc_group.setStyleSheet(Styles.group_box(Styles.COLORS['green']))
        acc_layout = QVBoxLayout(acc_group)
        
        if HAS_PYQTGRAPH:
            self._acc_plot = pg.PlotWidget()
            self._acc_plot.setBackground(Styles.COLORS['base'])
            self._acc_plot.showGrid(x=True, y=True, alpha=0.3)
            self._acc_plot.setLabel('left', 'Accuracy')
            self._acc_plot.setLabel('bottom', 'Epoch')
            acc_layout.addWidget(self._acc_plot)
        else:
            acc_layout.addWidget(QLabel("需要 pyqtgraph 显示图表"))
        
        left_layout.addWidget(acc_group)
        splitter.addWidget(left_widget)
        
        # 右侧：信息面板
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # 当前指标
        metrics_group = QGroupBox("当前指标")
        metrics_group.setStyleSheet(Styles.group_box(Styles.COLORS['blue']))
        metrics_layout = QVBoxLayout(metrics_group)
        
        self._metrics_table = QTableWidget()
        self._metrics_table.setColumnCount(2)
        self._metrics_table.setHorizontalHeaderLabels(["指标", "值"])
        self._metrics_table.horizontalHeader().setStretchLastSection(True)
        self._metrics_table.verticalHeader().setVisible(False)
        self._metrics_table.setStyleSheet(f"""
            QTableWidget {{
                background: {Styles.COLORS['surface0']};
                border: none;
                color: {Styles.COLORS['text']};
            }}
            QHeaderView::section {{
                background: {Styles.COLORS['surface1']};
                color: {Styles.COLORS['text']};
                padding: 6px;
                border: none;
            }}
        """)
        metrics_layout.addWidget(self._metrics_table)
        right_layout.addWidget(metrics_group)
        
        # 训练日志
        log_group = QGroupBox("训练日志")
        log_group.setStyleSheet(Styles.group_box(Styles.COLORS['purple']))
        log_layout = QVBoxLayout(log_group)
        
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setStyleSheet(f"""
            QTextEdit {{
                background: {Styles.COLORS['crust']};
                color: {Styles.COLORS['text']};
                border: none;
                font-family: Consolas, monospace;
                font-size: 12px;
            }}
        """)
        log_layout.addWidget(self._log_text)
        right_layout.addWidget(log_group)
        
        splitter.addWidget(right_widget)
        splitter.setSizes([600, 400])
    
    def _create_header(self) -> QWidget:
        """创建标题栏"""
        header = QFrame()
        header.setStyleSheet(f"""
            QFrame {{
                background: {Styles.COLORS['surface0']};
                border-radius: 6px;
            }}
        """)
        
        layout = QVBoxLayout(header)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)
        
        # 第一行：标题和状态
        row1 = QHBoxLayout()
        
        title = QLabel("🏋️ 训练监控")
        title.setStyleSheet(f"""
            font-size: 16px;
            font-weight: bold;
            color: {Styles.COLORS['text']};
        """)
        row1.addWidget(title)
        
        self._status_label = QLabel("空闲")
        self._status_label.setStyleSheet(f"""
            color: {Styles.COLORS['subtext1']};
            padding: 4px 12px;
            background: {Styles.COLORS['surface1']};
            border-radius: 10px;
        """)
        row1.addWidget(self._status_label)
        
        row1.addStretch()
        
        # 控制按钮
        self._pause_btn = QPushButton("⏸️ 暂停")
        self._pause_btn.clicked.connect(self._on_pause_resume)
        self._pause_btn.setEnabled(False)
        row1.addWidget(self._pause_btn)
        
        self._stop_btn = QPushButton("⏹️ 停止")
        self._stop_btn.clicked.connect(self._on_stop)
        self._stop_btn.setEnabled(False)
        row1.addWidget(self._stop_btn)
        
        for btn in [self._pause_btn, self._stop_btn]:
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: {Styles.COLORS['surface1']};
                    border: none;
                    border-radius: 4px;
                    padding: 6px 16px;
                    color: {Styles.COLORS['text']};
                }}
                QPushButton:hover {{
                    background: {Styles.COLORS['surface2']};
                }}
                QPushButton:disabled {{
                    color: {Styles.COLORS['overlay0']};
                }}
            """)
        
        layout.addLayout(row1)
        
        # 第二行：进度条
        row2 = QHBoxLayout()
        
        self._epoch_label = QLabel("Epoch: 0 / 0")
        self._epoch_label.setStyleSheet(f"color: {Styles.COLORS['subtext1']};")
        row2.addWidget(self._epoch_label)
        
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setStyleSheet(Styles.PROGRESS_BAR)
        row2.addWidget(self._progress_bar)
        
        self._time_label = QLabel("--:--")
        self._time_label.setStyleSheet(f"color: {Styles.COLORS['subtext1']};")
        row2.addWidget(self._time_label)
        
        layout.addLayout(row2)
        
        return header
    
    def start_training(self, total_epochs: int):
        """开始训练"""
        self._is_training = True
        self._is_paused = False
        
        # 清除历史数据
        self._epochs.clear()
        self._train_loss.clear()
        self._val_loss.clear()
        self._train_acc.clear()
        self._val_acc.clear()
        
        # 更新UI
        self._status_label.setText("训练中...")
        self._status_label.setStyleSheet(f"""
            color: {Styles.COLORS['base']};
            padding: 4px 12px;
            background: {Styles.COLORS['green']};
            border-radius: 10px;
        """)
        
        self._epoch_label.setText(f"Epoch: 0 / {total_epochs}")
        self._progress_bar.setRange(0, total_epochs)
        self._progress_bar.setValue(0)
        
        self._pause_btn.setEnabled(True)
        self._stop_btn.setEnabled(True)
        
        self._log_text.clear()
        self._log("训练开始...")
        
        # 清除图表
        if HAS_PYQTGRAPH:
            self._loss_plot.clear()
            self._acc_plot.clear()
    
    def update_epoch(self, epoch: int, total_epochs: int, metrics: Dict[str, float]):
        """
        更新epoch信息
        
        Args:
            epoch: 当前epoch
            total_epochs: 总epoch数
            metrics: 指标字典
        """
        self._epoch_label.setText(f"Epoch: {epoch} / {total_epochs}")
        self._progress_bar.setValue(epoch)
        
        # 记录历史
        self._epochs.append(epoch)
        
        if 'loss' in metrics:
            self._train_loss.append(metrics['loss'])
        if 'val_loss' in metrics:
            self._val_loss.append(metrics['val_loss'])
        if 'accuracy' in metrics:
            self._train_acc.append(metrics['accuracy'])
        if 'val_accuracy' in metrics:
            self._val_acc.append(metrics['val_accuracy'])
        
        # 更新图表
        self._update_plots()
        
        # 更新指标表格
        self._update_metrics_table(metrics)
        
        # 日志
        metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        self._log(f"Epoch {epoch}/{total_epochs}: {metrics_str}")
    
    def _update_plots(self):
        """更新图表"""
        if not HAS_PYQTGRAPH:
            return
        
        # 损失曲线
        self._loss_plot.clear()
        if self._train_loss:
            self._loss_plot.plot(
                self._epochs, self._train_loss,
                pen=pg.mkPen(color=Styles.COLORS['blue'], width=2),
                name='Train Loss'
            )
        if self._val_loss:
            self._loss_plot.plot(
                self._epochs[:len(self._val_loss)], self._val_loss,
                pen=pg.mkPen(color=Styles.COLORS['red'], width=2),
                name='Val Loss'
            )
        
        # 准确率曲线
        self._acc_plot.clear()
        if self._train_acc:
            self._acc_plot.plot(
                self._epochs, self._train_acc,
                pen=pg.mkPen(color=Styles.COLORS['green'], width=2),
                name='Train Acc'
            )
        if self._val_acc:
            self._acc_plot.plot(
                self._epochs[:len(self._val_acc)], self._val_acc,
                pen=pg.mkPen(color=Styles.COLORS['purple'], width=2),
                name='Val Acc'
            )
    
    def _update_metrics_table(self, metrics: Dict[str, float]):
        """更新指标表格"""
        self._metrics_table.setRowCount(len(metrics))
        
        for i, (name, value) in enumerate(metrics.items()):
            name_item = QTableWidgetItem(name)
            value_item = QTableWidgetItem(f"{value:.6f}")
            
            self._metrics_table.setItem(i, 0, name_item)
            self._metrics_table.setItem(i, 1, value_item)
    
    def finish_training(self, success: bool, message: str = ""):
        """完成训练"""
        self._is_training = False
        self._is_paused = False
        
        if success:
            self._status_label.setText("完成")
            self._status_label.setStyleSheet(f"""
                color: {Styles.COLORS['base']};
                padding: 4px 12px;
                background: {Styles.COLORS['blue']};
                border-radius: 10px;
            """)
            self._log(f"训练完成! {message}")
        else:
            self._status_label.setText("失败")
            self._status_label.setStyleSheet(f"""
                color: {Styles.COLORS['base']};
                padding: 4px 12px;
                background: {Styles.COLORS['red']};
                border-radius: 10px;
            """)
            self._log(f"训练失败: {message}")
        
        self._pause_btn.setEnabled(False)
        self._stop_btn.setEnabled(False)
    
    def _on_pause_resume(self):
        """暂停/恢复"""
        if self._is_paused:
            self._is_paused = False
            self._pause_btn.setText("⏸️ 暂停")
            self._status_label.setText("训练中...")
            self.resume_requested.emit()
        else:
            self._is_paused = True
            self._pause_btn.setText("▶️ 恢复")
            self._status_label.setText("已暂停")
            self.pause_requested.emit()
    
    def _on_stop(self):
        """停止"""
        self.stop_requested.emit()
    
    def _log(self, message: str):
        """添加日志"""
        self._log_text.append(message)
        # 滚动到底部
        scrollbar = self._log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def set_history(self, history: Dict[str, List[float]]):
        """设置训练历史（用于加载已完成的训练）"""
        self._train_loss = history.get('loss', [])
        self._val_loss = history.get('val_loss', [])
        self._train_acc = history.get('accuracy', [])
        self._val_acc = history.get('val_accuracy', [])
        self._epochs = list(range(1, len(self._train_loss) + 1))
        
        self._update_plots()
    
    def clear(self):
        """清除所有"""
        self._epochs.clear()
        self._train_loss.clear()
        self._val_loss.clear()
        self._train_acc.clear()
        self._val_acc.clear()
        
        self._status_label.setText("空闲")
        self._status_label.setStyleSheet(f"""
            color: {Styles.COLORS['subtext1']};
            padding: 4px 12px;
            background: {Styles.COLORS['surface1']};
            border-radius: 10px;
        """)
        
        self._epoch_label.setText("Epoch: 0 / 0")
        self._progress_bar.setValue(0)
        self._log_text.clear()
        self._metrics_table.setRowCount(0)
        
        if HAS_PYQTGRAPH:
            self._loss_plot.clear()
            self._acc_plot.clear()

