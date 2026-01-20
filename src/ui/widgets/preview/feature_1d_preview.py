# -*- coding: utf-8 -*-
"""
一维特征预览组件

显示一维特征的线图。
"""

import logging
from typing import Any, Optional

import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QGroupBox, QLabel, QVBoxLayout, QWidget
)

from src.workflow.port import DataType
from src.ui.styles import Styles

from .base_preview import BasePreviewWidget, register_preview


logger = logging.getLogger(__name__)


try:
    import pyqtgraph as pg
    from PyQt6.QtGui import QFont
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False


@register_preview
class Feature1DPreviewWidget(BasePreviewWidget):
    """
    一维特征预览组件
    
    显示一维特征数据的线图。
    """
    
    supported_types = [DataType.FEATURE_1D, DataType.FEATURE]
    display_name = "一维特征"
    icon = "📈"
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # 特征线图
        feature_group = QGroupBox("一维特征")
        feature_group.setStyleSheet(Styles.group_box(Styles.COLORS['teal']))
        feature_layout = QVBoxLayout(feature_group)
        feature_layout.setContentsMargins(4, 4, 4, 4)
        
        if HAS_PYQTGRAPH:
            self._plot_widget = pg.PlotWidget()
            self._plot_widget.setBackground('#181825')
            self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
            
            # 设置字体
            font = QFont("Microsoft YaHei", 9)
            for axis_name in ['left', 'bottom']:
                axis = self._plot_widget.getPlotItem().getAxis(axis_name)
                axis.setTickFont(font)
                axis.setStyle(tickTextOffset=5)
            
            self._plot_widget.setLabel('left', text='值')
            self._plot_widget.setLabel('bottom', text='索引')
            
            # 特征曲线
            self._curve = self._plot_widget.plot(
                pen=pg.mkPen(color='#94e2d5', width=2)
            )
            
            self._plot_widget.setMinimumHeight(400)
            feature_layout.addWidget(self._plot_widget)
        else:
            placeholder = QLabel("请安装 pyqtgraph 以显示特征图")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet(f"""
                background-color: {Styles.COLORS['mantle']};
                border: 1px dashed {Styles.COLORS['surface1']};
                border-radius: 4px;
                padding: 40px;
                color: {Styles.COLORS['subtext0']};
            """)
            feature_layout.addWidget(placeholder)
        
        layout.addWidget(feature_group)
        
        # 统计信息
        stats_group = QGroupBox("统计信息")
        stats_group.setStyleSheet(Styles.group_box(Styles.COLORS['lavender']))
        stats_layout = QVBoxLayout(stats_group)
        stats_layout.setContentsMargins(8, 8, 8, 8)
        
        self._stats_label = QLabel()
        self._stats_label.setStyleSheet(f"""
            QLabel {{
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 12px;
                color: {Styles.COLORS['text']};
                background: {Styles.COLORS['mantle']};
                padding: 8px;
                border-radius: 4px;
            }}
        """)
        stats_layout.addWidget(self._stats_label)
        layout.addWidget(stats_group)
    
    def set_data(self, data: Any) -> bool:
        """
        设置特征数据
        
        Args:
            data: FeatureData 对象或 numpy 数组
            
        Returns:
            是否成功设置
        """
        self._current_data = data
        
        if data is None:
            self.clear()
            return False
        
        try:
            # 获取实际数据数组
            if hasattr(data, 'data'):
                array = data.data
                feature_type = getattr(data, 'feature_type', 'unknown')
            elif isinstance(data, np.ndarray):
                array = data
                feature_type = 'array'
            else:
                logger.warning(f"不支持的数据类型: {type(data)}")
                return False
            
            # 确保是1D数据
            if array.ndim > 1:
                # 多维数据，展平或取均值
                if array.ndim == 2 and array.shape[0] == 1:
                    array = array[0]
                else:
                    array = array.flatten()
            
            # 更新数据信息
            self._update_data_info(
                f"📈 FEATURE_1D  shape={array.shape}  dtype={array.dtype}  type={feature_type}"
            )
            
            # 绘制曲线
            if HAS_PYQTGRAPH:
                x = np.arange(len(array))
                self._curve.setData(x, array)
                self._plot_widget.setXRange(0, len(array))
                
                # 自动调整Y轴范围
                if len(array) > 0:
                    y_min, y_max = np.min(array), np.max(array)
                    margin = (y_max - y_min) * 0.1 if y_max != y_min else 1
                    self._plot_widget.setYRange(y_min - margin, y_max + margin)
            
            # 更新统计信息
            self._update_stats(array)
            
            self.data_changed.emit()
            return True
            
        except Exception as e:
            logger.error(f"设置特征数据失败: {e}")
            return False
    
    def _update_stats(self, array: np.ndarray):
        """更新统计信息"""
        if len(array) == 0:
            self._stats_label.setText("无数据")
            return
        
        stats_text = (
            f"  长度: {len(array):,}\n"
            f"  最小值: {np.min(array):.6f}\n"
            f"  最大值: {np.max(array):.6f}\n"
            f"  均值: {np.mean(array):.6f}\n"
            f"  标准差: {np.std(array):.6f}\n"
            f"  中位数: {np.median(array):.6f}"
        )
        self._stats_label.setText(stats_text)
    
    def clear(self):
        """清除显示"""
        self._current_data = None
        self._data_info = ""
        if HAS_PYQTGRAPH:
            self._curve.clear()
        self._stats_label.setText("")
    
    @classmethod
    def can_display(cls, data: Any) -> bool:
        """检查是否可以显示该数据"""
        # 检查 FeatureData 且为1D
        if hasattr(data, 'data') and hasattr(data, 'feature_type'):
            arr = data.data
            # 1D 特征：形状为 (n,) 或 (1, n) 或 (channels, n) 且 n 较小
            if arr.ndim == 1:
                return True
            if arr.ndim == 2 and arr.shape[0] <= 2 and arr.shape[1] > arr.shape[0]:
                # 可能是 (channels, features) 的1D特征
                return True
            return False
        
        # 纯 numpy 1D 数组
        if isinstance(data, np.ndarray) and data.ndim == 1:
            return True
        
        return False

