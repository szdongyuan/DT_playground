# -*- coding: utf-8 -*-
"""
2D Feature Preview Component

Displays heatmaps of 2D features.
"""

import logging
from typing import Any, Optional

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox, QHBoxLayout, QLabel, QSplitter, QVBoxLayout, QWidget
)

from src.ui.i18n import tr_
from src.ui.styles import Styles

from .base_preview import BasePreviewWidget, register_preview


logger = logging.getLogger(__name__)


try:
    import pyqtgraph as pg
    from PySide6.QtGui import QFont
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False


@register_preview
class Feature2DPreviewWidget(BasePreviewWidget):
    """
    二维特征预览组件
    
    显示二维特征数据的热力图和边际分布。
    """
    
    display_name = tr_("2D feature")
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
        self._heatmap_group = QGroupBox(tr_("2D feature heatmap"))
        self._heatmap_group.setStyleSheet(Styles.group_box(Styles.COLORS['purple']))
        heatmap_layout = QVBoxLayout(self._heatmap_group)
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
            placeholder = QLabel(tr_("Please install pyqtgraph to display heatmaps"))
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet(f"""
                background-color: {Styles.COLORS['mantle']};
                border: 1px dashed {Styles.COLORS['surface1']};
                border-radius: 4px;
                padding: 40px;
                color: {Styles.COLORS['subtext0']};
            """)
            heatmap_layout.addWidget(placeholder)
        
        splitter.addWidget(self._heatmap_group)
        
        # 时间维度均值曲线
        curve_group = QGroupBox(tr_("Mean over time"))
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
            
            self._curve_widget.setLabel('left', text=tr_('Mean'))
            self._curve_widget.setLabel('bottom', text=tr_('Frame'))
            
            self._mean_curve = self._curve_widget.plot(
                pen=pg.mkPen(color='#89b4fa', width=2)
            )
            
            self._curve_widget.setMinimumHeight(150)
            curve_layout.addWidget(self._curve_widget)
        else:
            placeholder = QLabel(tr_("Please install pyqtgraph"))
            curve_layout.addWidget(placeholder)
        
        splitter.addWidget(curve_group)
        
        # 统计信息
        stats_group = QGroupBox(tr_("Statistics"))
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
                f"{self._feature_icon(feature_type)} FEATURE_2D  shape={array.shape}  ch={channels}  "
                f"dtype={array.dtype}  type={feature_type}"
            )
            
            if HAS_PYQTGRAPH:
                self._apply_colormap(feature_type)
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
            self._update_title(feature_type)
            self._update_stats(display_array, feature_type)
            
            self.data_changed.emit()
            return True
            
        except Exception as e:
            logger.error(f"设置特征数据失败: {e}")
            return False
    
    def _feature_icon(self, feature_type: str) -> str:
        if feature_type == "grad_cam":
            return "🔥"
        if feature_type == "grad_cam_overlay":
            return "🧩"
        return "🗺️"

    def _update_title(self, feature_type: str):
        if feature_type == "grad_cam":
            self._heatmap_group.setTitle(tr_("Grad-CAM heatmap"))
        elif feature_type == "grad_cam_overlay":
            self._heatmap_group.setTitle(tr_("Grad-CAM overlay"))
        else:
            self._heatmap_group.setTitle(tr_("2D feature heatmap"))

    def _apply_colormap(self, feature_type: str):
        if not HAS_PYQTGRAPH:
            return
        colormap_name = "viridis"
        if feature_type == "grad_cam":
            colormap_name = "inferno"
        elif feature_type == "grad_cam_overlay":
            colormap_name = "magma"
        try:
            self._image_view.setColorMap(pg.colormap.get(colormap_name))
        except Exception:
            self._image_view.setColorMap(pg.colormap.get("viridis"))

    def _update_stats(self, array: np.ndarray, feature_type: str = ""):
        """更新统计信息"""
        if array.size == 0:
            self._stats_label.setText(tr_("No data"))
            return
        note = ""
        if feature_type == "grad_cam":
            note = tr_("  Note: higher values indicate stronger class evidence.\n")
        elif feature_type == "grad_cam_overlay":
            note = tr_("  Note: normalized source feature blended with Grad-CAM intensity.\n")
        
        stats_text = (
            note
            + tr_("  Shape: {h} × {w} (features × frames)\n").format(h=array.shape[0], w=array.shape[1])
            + tr_("  Elements: {n:,}\n").format(n=int(array.size))
            + tr_("  Min: {value:.6f}\n").format(value=float(np.min(array)))
            + tr_("  Max: {value:.6f}\n").format(value=float(np.max(array)))
            + tr_("  Mean: {value:.6f}\n").format(value=float(np.mean(array)))
            + tr_("  Std: {value:.6f}").format(value=float(np.std(array)))
        )
        self._stats_label.setText(stats_text)
    
    def clear(self):
        """清除显示"""
        self._current_data = None
        self._data_info = ""
        self._heatmap_group.setTitle(tr_("2D feature heatmap"))
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

