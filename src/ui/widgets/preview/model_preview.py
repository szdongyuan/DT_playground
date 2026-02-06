# -*- coding: utf-8 -*-
"""
Model Preview Component

Displays Keras model summary information.
"""

import io
import logging
from typing import Any, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QGroupBox, QHBoxLayout, QLabel, QSplitter, 
    QTextEdit, QVBoxLayout, QWidget
)

from src.ui.i18n import tr_
from src.ui.styles import Styles

from .base_preview import BasePreviewWidget, register_preview


logger = logging.getLogger(__name__)


@register_preview
class ModelPreviewWidget(BasePreviewWidget):
    """
    模型预览组件
    
    显示 Keras 模型的 summary 信息和参数统计。
    """
    
    display_name = tr_("Model preview")
    icon = "🧠"
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # 主分割器
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)
        layout.addWidget(splitter)
        
        # 左侧：模型 Summary
        summary_group = QGroupBox(tr_("🧠 Model summary"))
        summary_group.setStyleSheet(Styles.group_box(Styles.COLORS['mauve']))
        summary_layout = QVBoxLayout(summary_group)
        summary_layout.setContentsMargins(8, 8, 8, 8)
        
        self._summary_text = QTextEdit()
        self._summary_text.setReadOnly(True)
        self._summary_text.setStyleSheet(f"""
            QTextEdit {{
                font-family: 'Consolas', 'Courier New', 'Monaco', monospace;
                font-size: 12px;
                color: {Styles.COLORS['text']};
                background: {Styles.COLORS['mantle']};
                border: 1px solid {Styles.COLORS['surface1']};
                border-radius: 6px;
                padding: 8px;
            }}
        """)
        summary_layout.addWidget(self._summary_text)
        
        splitter.addWidget(summary_group)
        
        # 右侧：参数统计
        stats_group = QGroupBox(tr_("📊 Model parameters"))
        stats_group.setStyleSheet(Styles.group_box(Styles.COLORS['teal']))
        stats_layout = QVBoxLayout(stats_group)
        stats_layout.setContentsMargins(12, 12, 12, 12)
        
        self._stats_label = QLabel()
        self._stats_label.setStyleSheet(f"""
            QLabel {{
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 14px;
                color: {Styles.COLORS['text']};
                background: {Styles.COLORS['mantle']};
                padding: 16px;
                border-radius: 8px;
                line-height: 2.0;
            }}
        """)
        self._stats_label.setWordWrap(True)
        self._stats_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        stats_layout.addWidget(self._stats_label)
        stats_layout.addStretch()
        
        splitter.addWidget(stats_group)
        
        # 设置分割器比例
        splitter.setSizes([500, 250])
    
    def set_data(self, data: Any) -> bool:
        """
        设置模型数据
        
        Args:
            data: Keras 模型对象
            
        Returns:
            是否成功设置
        """
        self._current_data = data
        
        if data is None:
            self.clear()
            return False
        
        try:
            # 捕获 model.summary() 输出
            summary_str = self._get_model_summary(data)
            
            # 获取参数统计
            stats = self._get_model_stats(data)
            
            # 更新数据信息
            model_name = getattr(data, 'name', 'Unknown')
            total_params = stats.get('total_params', 0)
            self._update_data_info(
                f"🧠 MODEL  name={model_name}  params={self._format_params(total_params)}"
            )
            
            # 更新 Summary 显示
            self._summary_text.setPlainText(summary_str)
            
            # 更新统计信息
            self._update_stats(stats)
            
            # 打印到日志
            self._log_summary(model_name, summary_str, stats)
            
            self.data_changed.emit()
            return True
            
        except Exception as e:
            logger.error(f"设置模型数据失败: {e}")
            self._summary_text.setPlainText(
                tr_("Failed to parse model: {error}").format(error=str(e))
            )
            return False
    
    def _get_model_summary(self, model) -> str:
        """
        获取模型的 summary 字符串
        
        使用 io.StringIO 捕获 model.summary() 的输出。
        """
        try:
            # 使用 StringIO 捕获 summary 输出
            stream = io.StringIO()
            model.summary(print_fn=lambda x: stream.write(x + '\n'))
            return stream.getvalue()
        except Exception as e:
            logger.warning(f"获取模型 summary 失败: {e}")
            return tr_("Failed to get model summary: {error}").format(error=str(e))
    
    def _get_model_stats(self, model) -> dict:
        """获取模型参数统计信息"""
        stats = {}
        
        try:
            # 模型名称
            stats['name'] = getattr(model, 'name', 'Unknown')
            
            # 参数统计
            if hasattr(model, 'count_params'):
                stats['total_params'] = model.count_params()
            
            # 可训练与不可训练参数
            if hasattr(model, 'trainable_variables') and hasattr(model, 'non_trainable_variables'):
                import tensorflow as tf
                trainable_count = sum(
                    tf.reduce_prod(v.shape).numpy() 
                    for v in model.trainable_variables
                )
                non_trainable_count = sum(
                    tf.reduce_prod(v.shape).numpy() 
                    for v in model.non_trainable_variables
                )
                stats['trainable_params'] = int(trainable_count)
                stats['non_trainable_params'] = int(non_trainable_count)
            
            # 层数
            if hasattr(model, 'layers'):
                stats['num_layers'] = len(model.layers)
            
            # 输入形状
            if hasattr(model, 'input_shape'):
                stats['input_shape'] = str(model.input_shape)
            
            # 输出形状
            if hasattr(model, 'output_shape'):
                stats['output_shape'] = str(model.output_shape)
            
            # 是否已编译
            if hasattr(model, 'compiled'):
                stats['compiled'] = model.compiled
            
            # 优化器信息
            if hasattr(model, 'optimizer') and model.optimizer:
                optimizer = model.optimizer
                if hasattr(optimizer, 'get_config'):
                    config = optimizer.get_config()
                    stats['optimizer'] = config.get('name', str(type(optimizer).__name__))
                    stats['learning_rate'] = config.get('learning_rate', 'N/A')
                else:
                    stats['optimizer'] = str(type(optimizer).__name__)
            
            # 损失函数
            if hasattr(model, 'loss') and model.loss:
                if isinstance(model.loss, str):
                    stats['loss'] = model.loss
                elif hasattr(model.loss, '__name__'):
                    stats['loss'] = model.loss.__name__
                else:
                    stats['loss'] = str(model.loss)
                    
        except Exception as e:
            logger.warning(f"获取模型统计信息失败: {e}")
        
        return stats
    
    def _update_stats(self, stats: dict):
        """更新统计信息显示"""
        lines = []
        lines.append("═" * 35)
        lines.append("")
        
        # 基本信息
        if 'name' in stats:
            lines.append(f"  <b>{tr_('Model name')}:</b>  {stats['name']}")
        
        if 'num_layers' in stats:
            lines.append(f"  <b>{tr_('Layers')}:</b>  {stats['num_layers']}")
        
        lines.append("")
        lines.append("─" * 35)
        lines.append("")
        
        # 参数统计
        if 'total_params' in stats:
            lines.append(
                f"  <b>{tr_('Total params')}:</b>  {self._format_params(stats['total_params'])}"
            )
        
        if 'trainable_params' in stats:
            lines.append(
                f"  <b>{tr_('Trainable params')}:</b>  {self._format_params(stats['trainable_params'])}"
            )
        
        if 'non_trainable_params' in stats:
            lines.append(
                f"  <b>{tr_('Non-trainable params')}:</b>  {self._format_params(stats['non_trainable_params'])}"
            )
        
        lines.append("")
        lines.append("─" * 35)
        lines.append("")
        
        # 形状信息
        if 'input_shape' in stats:
            lines.append(f"  <b>{tr_('Input shape')}:</b>  {stats['input_shape']}")
        
        if 'output_shape' in stats:
            lines.append(f"  <b>{tr_('Output shape')}:</b>  {stats['output_shape']}")
        
        lines.append("")
        lines.append("─" * 35)
        lines.append("")
        
        # 编译信息
        if 'compiled' in stats:
            status = tr_("✅ Compiled") if stats['compiled'] else tr_("❌ Not compiled")
            lines.append(f"  <b>{tr_('Compile status')}:</b>  {status}")
        
        if 'optimizer' in stats:
            lines.append(f"  <b>{tr_('Optimizer')}:</b>  {stats['optimizer']}")
        
        if 'learning_rate' in stats:
            lr = stats['learning_rate']
            if isinstance(lr, float):
                lines.append(f"  <b>{tr_('Learning rate')}:</b>  {lr:.6f}")
            else:
                lines.append(f"  <b>{tr_('Learning rate')}:</b>  {lr}")
        
        if 'loss' in stats:
            lines.append(f"  <b>{tr_('Loss')}:</b>  {stats['loss']}")
        
        lines.append("")
        lines.append("═" * 35)
        
        self._stats_label.setText("<br>".join(lines))
    
    def _format_params(self, count: int) -> str:
        """格式化参数数量"""
        if count >= 1_000_000_000:
            return f"{count / 1_000_000_000:.2f}B ({count:,})"
        elif count >= 1_000_000:
            return f"{count / 1_000_000:.2f}M ({count:,})"
        elif count >= 1_000:
            return f"{count / 1_000:.2f}K ({count:,})"
        else:
            return f"{count:,}"
    
    def _log_summary(self, model_name: str, summary: str, stats: dict):
        """打印模型 summary 到日志"""
        logger.info("=" * 60)
        logger.info(f"🧠 Model summary: {model_name}")
        logger.info("=" * 60)
        
        # 打印 summary
        for line in summary.split('\n'):
            if line.strip():
                logger.info(line)
        
        logger.info("-" * 60)
        logger.info("📊 Parameter stats:")
        
        if 'total_params' in stats:
            logger.info(f"  Total params: {self._format_params(stats['total_params'])}")
        if 'trainable_params' in stats:
            logger.info(f"  Trainable params: {self._format_params(stats['trainable_params'])}")
        if 'non_trainable_params' in stats:
            logger.info(f"  Non-trainable params: {self._format_params(stats['non_trainable_params'])}")
        
        logger.info("=" * 60)
    
    def clear(self):
        """清除显示"""
        self._current_data = None
        self._data_info = ""
        self._summary_text.clear()
        self._stats_label.setText("")
    
    @classmethod
    def can_display(cls, data: Any) -> bool:
        """检查是否可以显示该数据"""
        # 检查是否是 Keras 模型
        try:
            from tensorflow import keras
            return isinstance(data, keras.Model)
        except ImportError:
            pass
        
        # 备用检查：检查是否有 summary 方法和 layers 属性
        if hasattr(data, 'summary') and hasattr(data, 'layers'):
            return True
        
        return False
