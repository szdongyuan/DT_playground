# -*- coding: utf-8 -*-
"""
2D Feature Preview Component

Displays heatmaps of 2D features.
"""

import logging
from typing import Any, Optional

import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QGroupBox, QHBoxLayout, QLabel, QSplitter, QVBoxLayout, QWidget
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
class Feature2DPreviewWidget(BasePreviewWidget):
    """
    二维特征预览组件
    
    显示二维特征数据的热力图和边际分布。
    """
    
    supported_types = [DataType.FEATURE_2D, DataType.FEATURE]
    display_name = "二维特征"
    icon = "🗺️"
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # 主分割器
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setHandleWidth(2)
        layout.addWidget(splitter)
        
        # 热力图
        heatmap_group = QGroupBox("二维特征热力图")
        heatmap_group.setStyleSheet(Styles.group_box(Styles.COLORS['purple']))
        heatmap_layout = QVBoxLayout(heatmap_group)
        heatmap_layout.setContentsMargins(4, 4, 4, 4)
        
        if HAS_PYQTGRAPH:
            self._image_view = pg.ImageView()
            self._image_view.ui.histogram.hide()
            self._image_view.ui.roiBtn.hide()
            self._image_view.ui.menuBtn.hide()
            self._image_view.view.setBackgroundColor('#181825')
            
            # 设置 y 轴方向：低频在下，高频在上
            self._image_view.view.invertY(False)
            self._image_view.view.setAspectLocked(False)
            
            # 设置颜色映射
            colormap = pg.colormap.get('viridis')
            self._image_view.setColorMap(colormap)
            
            self._image_view.setMinimumHeight(300)
            heatmap_layout.addWidget(self._image_view)
        else:
            placeholder = QLabel("请安装 pyqtgraph 以显示热力图")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet(f"""
                background-color: {Styles.COLORS['mantle']};
                border: 1px dashed {Styles.COLORS['surface1']};
                border-radius: 4px;
                padding: 40px;
                color: {Styles.COLORS['subtext0']};
            """)
            heatmap_layout.addWidget(placeholder)
        
        splitter.addWidget(heatmap_group)
        
        # 时间维度均值曲线
        curve_group = QGroupBox("时间维度均值")
        curve_group.setStyleSheet(Styles.group_box(Styles.COLORS['blue']))
        curve_layout = QVBoxLayout(curve_group)
        curve_layout.setContentsMargins(4, 4, 4, 4)
        
        if HAS_PYQTGRAPH:
            self._curve_widget = pg.PlotWidget()
            self._curve_widget.setBackground('#181825')
            self._curve_widget.showGrid(x=True, y=True, alpha=0.3)
            
            font = QFont("Microsoft YaHei", 9)
            for axis_name in ['left', 'bottom']:
                axis = self._curve_widget.getPlotItem().getAxis(axis_name)
                axis.setTickFont(font)
            
            self._curve_widget.setLabel('left', text='均值')
            self._curve_widget.setLabel('bottom', text='时间帧')
            
            self._mean_curve = self._curve_widget.plot(
                pen=pg.mkPen(color='#89b4fa', width=2)
            )
            
            self._curve_widget.setMinimumHeight(150)
            curve_layout.addWidget(self._curve_widget)
        else:
            placeholder = QLabel("请安装 pyqtgraph")
            curve_layout.addWidget(placeholder)
        
        splitter.addWidget(curve_group)
        
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
        
        # 设置分割器比例
        splitter.setSizes([300, 150])
    
    def set_data(self, data: Any) -> bool:
        """
        设置特征数据
        
        Args:
            data: FeatureData 对象或 2D numpy 数组
            
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
                channels = getattr(data, 'channels', 1)
            elif isinstance(data, np.ndarray):
                array = data
                feature_type = 'array'
                channels = 1
            else:
                logger.warning(f"不支持的数据类型: {type(data)}")
                return False
            
            # 处理多通道数据：取第一个通道
            if array.ndim == 3:
                # (channels, features, frames)
                display_array = array[0]
            elif array.ndim == 2:
                display_array = array
            else:
                logger.warning(f"不支持的数组维度: {array.ndim}")
                return False
            
            # 更新数据信息
            self._update_data_info(
                f"🗺️ FEATURE_2D  shape={array.shape}  ch={channels}  "
                f"dtype={array.dtype}  type={feature_type}"
            )
            
            if HAS_PYQTGRAPH:
                # 显示热力图
                # 转置使时间为x轴，频率为y轴
                self._image_view.setImage(
                    display_array.T, autoRange=True, autoLevels=True
                )
                
                # 显示时间维度均值曲线
                mean_curve = np.mean(display_array, axis=0)
                x = np.arange(len(mean_curve))
                self._mean_curve.setData(x, mean_curve)
                self._curve_widget.setXRange(0, len(mean_curve))
            
            # 更新统计信息
            self._update_stats(display_array)
            
            self.data_changed.emit()
            return True
            
        except Exception as e:
            logger.error(f"设置特征数据失败: {e}")
            return False
    
    def _update_stats(self, array: np.ndarray):
        """更新统计信息"""
        if array.size == 0:
            self._stats_label.setText("无数据")
            return
        
        stats_text = (
            f"  形状: {array.shape[0]} × {array.shape[1]} (特征 × 时间帧)\n"
            f"  总元素: {array.size:,}\n"
            f"  最小值: {np.min(array):.6f}\n"
            f"  最大值: {np.max(array):.6f}\n"
            f"  均值: {np.mean(array):.6f}\n"
            f"  标准差: {np.std(array):.6f}"
        )
        self._stats_label.setText(stats_text)
    
    def clear(self):
        """清除显示"""
        self._current_data = None
        self._data_info = ""
        if HAS_PYQTGRAPH:
            self._image_view.clear()
            self._mean_curve.clear()
        self._stats_label.setText("")
    
    @classmethod
    def can_display(cls, data: Any) -> bool:
        """检查是否可以显示该数据"""
        # 检查 FeatureData 且为2D
        if hasattr(data, 'data') and hasattr(data, 'feature_type'):
            arr = data.data
            # 2D 特征：形状为 (features, frames) 或 (channels, features, frames)
            if arr.ndim >= 2:
                # 排除1D特征的情况
                if arr.ndim == 2 and arr.shape[0] <= 2 and arr.shape[1] > 100:
                    # 可能是1D特征
                    return False
                return True
            return False
        
        # 纯 numpy 2D 数组
        if isinstance(data, np.ndarray) and data.ndim == 2:
            # 确保不是太扁的数组（那可能是1D特征）
            if data.shape[0] > 3:
                return True
        
        return False

