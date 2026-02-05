# -*- coding: utf-8 -*-
"""
Layer Property Panel

Displays and edits parameters of the selected layer.
"""

import logging
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFrame,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget
)

from src.model_builder.layer_base import LayerNode, LayerParameter
from src.ui.i18n import tr_
from src.ui.styles import Styles

logger = logging.getLogger(__name__)


class LayerPropertyPanel(QWidget):
    """
    层属性面板
    
    显示和编辑选中层的参数。
    编译配置现在位于 Output 输出层中。
    支持显示权重状态和冻结控制。
    """
    
    parameter_changed = pyqtSignal(str, str, object)  # layer_id, param_name, value
    compile_config_changed = pyqtSignal()  # 编译配置变化信号（保留兼容性）
    trainable_changed = pyqtSignal(str, bool)  # layer_id, trainable 冻结状态变化信号
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_layer: Optional[LayerNode] = None
        self._widgets: Dict[str, QWidget] = {}
        self._freeze_widgets: Dict[str, QWidget] = {}  # 冻结控制相关控件
        
        self._setup_ui()
        self._apply_styles()
    
    def _setup_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # 标题
        header = QFrame()
        header.setFixedHeight(40)
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(12, 8, 12, 8)
        
        self._title = QLabel(tr_("⚙️ Properties"))
        self._title.setStyleSheet(f"""
            font-size: 14px;
            font-weight: bold;
            color: {Styles.COLORS['text']};
        """)
        header_layout.addWidget(self._title)
        layout.addWidget(header)
        
        # 滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(12, 8, 12, 8)
        self._content_layout.setSpacing(8)
        
        # 层属性标题
        self._layer_title = QLabel(tr_("📦 Layer Properties"))
        self._layer_title.setStyleSheet(f"""
            font-size: 13px;
            font-weight: bold;
            color: {Styles.COLORS['text']};
            margin-top: 8px;
        """)
        self._content_layout.addWidget(self._layer_title)
        
        # 层属性容器（动态内容放在这里，方便清除）
        self._layer_props_container = QWidget()
        self._layer_props_layout = QVBoxLayout(self._layer_props_container)
        self._layer_props_layout.setContentsMargins(0, 0, 0, 0)
        self._layer_props_layout.setSpacing(8)
        self._content_layout.addWidget(self._layer_props_container)
        
        # 无选中提示
        self._empty_label = QLabel(
            tr_(
                "Select a layer to edit its properties.\n\n"
                "Tip: compile config (optimizer, loss, etc.)\n"
                "is set in the Output layer."
            )
        )
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet(f"""
            color: {Styles.COLORS['subtext0']};
            padding: 20px;
        """)
        self._layer_props_layout.addWidget(self._empty_label)
        
        self._content_layout.addStretch()
        
        scroll.setWidget(self._content)
        layout.addWidget(scroll)
    
    def set_compile_config(self, config):
        """设置编译配置（保留接口兼容性，但不再使用）"""
        # 编译配置现在在 Output 层中，此方法保留以兼容旧代码
        pass
    
    def set_layer(self, layer: Optional[LayerNode]):
        """设置当前层"""
        self._current_layer = layer
        self._clear_widgets()
        
        if not layer:
            self._empty_label.show()
            self._layer_title.setText(tr_("📦 Layer Properties"))
            return
        
        self._empty_label.hide()
        self._layer_title.setText(f"📦 {layer.icon} {tr_(layer.display_name)}")
        
        # 层信息
        info_group = QGroupBox(tr_("Layer Info"))
        info_layout = QVBoxLayout(info_group)
        
        # 层名称
        name_layout = QVBoxLayout()
        name_label = QLabel(tr_("Layer name:"))
        name_input = QLineEdit(layer.layer_name)
        name_input.setPlaceholderText(tr_("Optional, for identifying the layer"))
        name_input.textChanged.connect(lambda v: self._on_name_changed(v))
        name_layout.addWidget(name_label)
        name_layout.addWidget(name_input)
        info_layout.addLayout(name_layout)
        
        # 层类型
        type_layout = QHBoxLayout()
        type_label = QLabel(tr_("Type:"))
        type_value = QLabel(layer.layer_type)
        type_value.setStyleSheet(f"color: {Styles.COLORS['subtext0']};")
        type_layout.addWidget(type_label)
        type_layout.addWidget(type_value)
        type_layout.addStretch()
        info_layout.addLayout(type_layout)
        
        self._layer_props_layout.addWidget(info_group)
        
        # 参数组
        if layer.parameters:
            params_group = QGroupBox(tr_("Parameters"))
            params_layout = QVBoxLayout(params_group)
            
            for name, param in layer.parameters.items():
                # multichoice 类型使用垂直布局
                if param.param_type == "multichoice":
                    widget = self._create_multichoice_widget(name, param, layer.parameter_values.get(name))
                    if widget:
                        label = QLabel(f"{param.display_name}:")
                        label.setToolTip(param.description)
                        params_layout.addWidget(label)
                        params_layout.addWidget(widget)
                        self._widgets[name] = widget
                else:
                    widget = self._create_param_widget(name, param, layer.parameter_values.get(name))
                    if widget:
                        # 使用垂直布局，标签在上
                        param_container = QVBoxLayout()
                        label = QLabel(f"{param.display_name}:")
                        label.setToolTip(param.description)
                        param_container.addWidget(label)
                        param_container.addWidget(widget)
                        params_layout.addLayout(param_container)
                        self._widgets[name] = widget
            
            self._layer_props_layout.addWidget(params_group)
        
        # 训练控制区域（权重状态和冻结控制）
        freeze_group = self._create_freeze_section(layer)
        self._layer_props_layout.addWidget(freeze_group)
    
    def _create_param_widget(self, name: str, param: LayerParameter, value: Any) -> Optional[QWidget]:
        """创建参数编辑控件"""
        widget = None
        
        if param.param_type == "int":
            widget = QSpinBox()
            widget.setRange(param.min_value or -999999, param.max_value or 999999)
            widget.setValue(value if value is not None else param.default_value)
            widget.valueChanged.connect(lambda v: self._on_param_changed(name, v))
            
        elif param.param_type == "float":
            widget = QDoubleSpinBox()
            widget.setRange(param.min_value or -999999, param.max_value or 999999)
            widget.setDecimals(6)
            widget.setSingleStep(0.0001)
            widget.setValue(value if value is not None else param.default_value)
            widget.valueChanged.connect(lambda v: self._on_param_changed(name, v))
            
        elif param.param_type == "bool":
            widget = QCheckBox()
            widget.setChecked(bool(value if value is not None else param.default_value))
            widget.stateChanged.connect(lambda v: self._on_param_changed(name, v == 2))
            
        elif param.param_type == "choice":
            widget = QComboBox()
            if param.choices:
                widget.addItems([str(c) for c in param.choices])
                current_value = str(value if value is not None else param.default_value)
                idx = widget.findText(current_value)
                if idx >= 0:
                    widget.setCurrentIndex(idx)
            widget.currentTextChanged.connect(lambda v: self._on_param_changed(name, v))
            
        elif param.param_type in ("str", "tuple", "list"):
            widget = QLineEdit()
            widget.setText(str(value if value is not None else param.default_value))
            widget.textChanged.connect(lambda v: self._on_param_changed(name, v))
        
        return widget
    
    def _create_multichoice_widget(self, name: str, param: LayerParameter, value: Any) -> Optional[QWidget]:
        """创建多选复选框控件"""
        if not param.choices:
            return None
        
        container = QFrame()
        container.setStyleSheet(f"""
            QFrame {{
                background: {Styles.COLORS['surface0']};
                border: 1px solid {Styles.COLORS['surface1']};
                border-radius: 6px;
                padding: 4px;
            }}
        """)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        
        # 获取当前选中的值
        current_values = value if value is not None else param.default_value
        if isinstance(current_values, str):
            current_values = [v.strip() for v in current_values.split(",") if v.strip()]
        elif not isinstance(current_values, list):
            current_values = [current_values] if current_values else []
        
        # 创建复选框字典（用于后续获取选中状态）
        checkboxes = {}
        
        # 每行放2个复选框
        row_layout = None
        for i, choice in enumerate(param.choices):
            if i % 2 == 0:
                row_layout = QHBoxLayout()
                row_layout.setSpacing(8)
                layout.addLayout(row_layout)
            
            cb = QCheckBox(str(choice))
            cb.setChecked(str(choice) in current_values)
            cb.setStyleSheet(f"""
                QCheckBox {{
                    color: {Styles.COLORS['text']};
                    padding: 2px;
                }}
            """)
            
            # 绑定变化事件
            cb.stateChanged.connect(
                lambda state, n=name, cbs=checkboxes: self._on_multichoice_changed(n, cbs)
            )
            
            checkboxes[str(choice)] = cb
            row_layout.addWidget(cb)
        
        # 补齐最后一行
        if row_layout and len(param.choices) % 2 == 1:
            row_layout.addStretch()
        
        # 保存复选框引用
        container.checkboxes = checkboxes
        
        return container
    
    def _on_multichoice_changed(self, name: str, checkboxes: Dict[str, QCheckBox]):
        """多选参数值变化"""
        if self._current_layer:
            selected = [choice for choice, cb in checkboxes.items() if cb.isChecked()]
            if not selected:
                selected = ["mae"]  # 至少保留一个默认值
            self._current_layer.set_parameter(name, selected)
            self.parameter_changed.emit(self._current_layer.layer_id, name, selected)
    
    def _create_freeze_section(self, layer: LayerNode) -> QGroupBox:
        """创建训练控制区域（权重状态和冻结控制）"""
        freeze_group = QGroupBox(tr_("Training Control"))
        freeze_layout = QVBoxLayout(freeze_group)
        freeze_layout.setSpacing(8)
        
        # 权重状态
        weight_layout = QHBoxLayout()
        weight_label = QLabel(tr_("Weights:"))
        weight_layout.addWidget(weight_label)
        
        if layer.has_weights:
            weight_status = QLabel(tr_("✅ Loaded"))
            weight_status.setStyleSheet(f"color: {Styles.COLORS['green']}; font-weight: bold;")
        else:
            weight_status = QLabel(tr_("⚪ None"))
            weight_status.setStyleSheet(f"color: {Styles.COLORS['subtext0']};")
        
        weight_layout.addWidget(weight_status)
        weight_layout.addStretch()
        freeze_layout.addLayout(weight_layout)
        
        # 可训练状态（冻结控制）
        trainable_layout = QHBoxLayout()
        trainable_label = QLabel(tr_("Trainable:"))
        trainable_layout.addWidget(trainable_label)
        
        trainable_checkbox = QCheckBox(tr_("Enabled"))
        trainable_checkbox.setChecked(layer.trainable)
        trainable_checkbox.setToolTip(
            tr_("Uncheck to freeze this layer so its weights won't be updated during training.")
        )
        trainable_checkbox.stateChanged.connect(
            lambda state: self._on_trainable_changed(state == 2)
        )
        trainable_layout.addWidget(trainable_checkbox)
        trainable_layout.addStretch()
        freeze_layout.addLayout(trainable_layout)
        
        # 冻结状态提示
        if not layer.trainable:
            frozen_hint = QLabel(tr_("🔒 This layer is frozen; weights will not be updated during training."))
            frozen_hint.setStyleSheet(f"""
                color: {Styles.COLORS['peach']};
                font-size: 11px;
                padding: 4px;
                background: {Styles.COLORS['surface1']};
                border-radius: 4px;
            """)
            freeze_layout.addWidget(frozen_hint)
        
        # 保存控件引用
        self._freeze_widgets['weight_status'] = weight_status
        self._freeze_widgets['trainable_checkbox'] = trainable_checkbox
        
        return freeze_group
    
    def _on_trainable_changed(self, trainable: bool):
        """可训练状态变化"""
        if self._current_layer:
            self._current_layer.trainable = trainable
            self.trainable_changed.emit(self._current_layer.layer_id, trainable)
            self.parameter_changed.emit(self._current_layer.layer_id, "_trainable", trainable)
            
            # 刷新显示以更新冻结提示
            layer = self._current_layer
            self.set_layer(None)
            self.set_layer(layer)
    
    def _on_name_changed(self, value: str):
        """层名称变化"""
        if self._current_layer:
            self._current_layer.layer_name = value
            self.parameter_changed.emit(self._current_layer.layer_id, "_name", value)
    
    def _on_param_changed(self, name: str, value: Any):
        """参数值变化"""
        if self._current_layer:
            self._current_layer.set_parameter(name, value)
            self.parameter_changed.emit(self._current_layer.layer_id, name, value)
    
    def _clear_widgets(self):
        """清除所有层属性控件（保留_empty_label）"""
        # 清除层属性容器中的所有widget，但保留_empty_label
        while self._layer_props_layout.count() > 0:
            item = self._layer_props_layout.takeAt(0)
            if item.widget() and item.widget() is not self._empty_label:
                item.widget().deleteLater()
        
        # 重新添加_empty_label
        self._layer_props_layout.addWidget(self._empty_label)
        self._widgets.clear()
        self._freeze_widgets.clear()
    
    def _apply_styles(self):
        """应用样式"""
        self.setStyleSheet(f"""
            LayerPropertyPanel {{
                background: {Styles.COLORS['mantle']};
                border-left: 1px solid {Styles.COLORS['surface0']};
            }}
            
            QScrollArea {{
                background: transparent;
                border: none;
            }}
            
            QGroupBox {{
                font-weight: bold;
                border: 1px solid {Styles.COLORS['surface1']};
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 12px;
                background: {Styles.COLORS['surface0']};
            }}
            
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: {Styles.COLORS['blue']};
            }}
            
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
                background: {Styles.COLORS['base']};
                border: 1px solid {Styles.COLORS['surface1']};
                border-radius: 4px;
                padding: 4px 8px;
                color: {Styles.COLORS['text']};
            }}
            
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
                border-color: {Styles.COLORS['blue']};
            }}
            
            QCheckBox {{
                color: {Styles.COLORS['text']};
            }}
            
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
                border-radius: 4px;
                border: 1px solid {Styles.COLORS['surface1']};
                background: {Styles.COLORS['base']};
            }}
            
            QCheckBox::indicator:checked {{
                background: {Styles.COLORS['blue']};
                border-color: {Styles.COLORS['blue']};
            }}
        """)

