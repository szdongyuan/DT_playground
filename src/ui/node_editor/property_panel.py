# -*- coding: utf-8 -*-
"""
Property Panel

Displays and edits node properties.
"""

import logging
import os
import sys
from typing import Any, Dict, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QFrame, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QScrollArea, QSpinBox,
    QVBoxLayout, QWidget
)

from ..i18n import tr_
from ..styles import Styles
from ...workflow.node_base import BaseNode, NodeParameter
from ...workflow.workflow import Workflow


logger = logging.getLogger(__name__)


def _get_app_root():
    """获取应用程序根目录（兼容 PyInstaller 打包）"""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        # 开发模式：向上寻找项目根目录
        current = os.path.dirname(os.path.abspath(__file__))
        # 从 src/ui/node_editor 向上三级到项目根目录
        return os.path.dirname(os.path.dirname(os.path.dirname(current)))


def _resolve_default_directory(default_dir: str) -> str:
    """将相对默认目录解析为绝对路径"""
    if not default_dir:
        return ""
    if os.path.isabs(default_dir):
        return default_dir
    # 相对路径基于应用程序根目录
    resolved = os.path.join(_get_app_root(), default_dir)
    return resolved


class PropertyPanel(QWidget):
    """
    属性面板
    
    显示选中节点的属性，允许编辑参数。
    
    Signals:
        parameter_changed: 参数值改变 (node_id, param_name, value)
    """
    
    parameter_changed = pyqtSignal(str, str, object)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._current_node: Optional[BaseNode] = None
        self._current_node_id: Optional[str] = None
        self._widgets: Dict[str, QWidget] = {}
        
        self._setup_ui()
    
    def _setup_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # 标题
        self._title_label = QLabel(tr_("Properties"))
        self._title_label.setStyleSheet(f"""
            QLabel {{
                font-size: 14px;
                font-weight: bold;
                color: {Styles.COLORS['text']};
                padding: 8px;
                background: {Styles.COLORS['surface0']};
                border-bottom: 1px solid {Styles.COLORS['surface1']};
            }}
        """)
        layout.addWidget(self._title_label)
        
        # 滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet(f"""
            QScrollArea {{
                background: {Styles.COLORS['base']};
                border: none;
            }}
        """)
        layout.addWidget(scroll)
        
        # 内容容器
        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(8, 8, 8, 8)
        self._content_layout.setSpacing(8)
        scroll.setWidget(self._content)
        
        # 节点信息区
        self._info_group = QGroupBox(tr_("Node Info"))
        self._info_group.setStyleSheet(Styles.group_box(Styles.COLORS['blue']))
        info_layout = QFormLayout(self._info_group)
        
        self._name_label = QLabel("-")
        self._type_label = QLabel("-")
        self._category_label = QLabel("-")
        self._desc_label = QLabel("-")
        self._desc_label.setWordWrap(True)
        
        info_layout.addRow(tr_("Name:"), self._name_label)
        info_layout.addRow(tr_("Type:"), self._type_label)
        info_layout.addRow(tr_("Category:"), self._category_label)
        info_layout.addRow(tr_("Description:"), self._desc_label)
        
        self._content_layout.addWidget(self._info_group)
        
        # 参数区
        self._params_group = QGroupBox(tr_("Parameters"))
        self._params_group.setStyleSheet(Styles.group_box(Styles.COLORS['green']))
        self._params_layout = QFormLayout(self._params_group)
        self._content_layout.addWidget(self._params_group)
        
        # 端口信息区
        self._ports_group = QGroupBox(tr_("Ports"))
        self._ports_group.setStyleSheet(Styles.group_box(Styles.COLORS['purple']))
        self._ports_layout = QVBoxLayout(self._ports_group)
        self._content_layout.addWidget(self._ports_group)
        
        # 添加弹性空间
        self._content_layout.addStretch()
        
        # 初始状态
        self._show_empty_state()
    
    def _show_empty_state(self):
        """显示空状态"""
        self._title_label.setText(tr_("Properties"))
        self._info_group.hide()
        self._params_group.hide()
        self._ports_group.hide()
    
    def set_node(self, node: BaseNode, node_id: str):
        """
        设置要显示的节点
        
        Args:
            node: 节点对象
            node_id: 节点ID
        """
        self._current_node = node
        self._current_node_id = node_id
        self._widgets.clear()
        
        if not node:
            self._show_empty_state()
            return
        
        # 更新标题
        self._title_label.setText(
            tr_("Properties - {name}").format(name=tr_(node.display_name))
        )
        
        # 显示节点信息
        self._info_group.show()
        self._name_label.setText(f"{node.icon} {tr_(node.display_name)}")
        self._type_label.setText(node.node_type)
        self._category_label.setText(node.category.display_name)
        self._desc_label.setText(tr_(node.description) if node.description else tr_("No description"))
        
        # 更新参数
        self._update_parameters(node)
        
        # 更新端口信息
        self._update_ports(node)
    
    def _update_parameters(self, node: BaseNode):
        """更新参数区域"""
        # 清除旧控件
        while self._params_layout.count():
            item = self._params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        if not node.parameters:
            self._params_group.hide()
            return
        
        self._params_group.show()
        
        for param_name, param in node.parameters.items():
            widget = self._create_param_widget(param, node.parameter_values.get(param_name))
            if widget:
                self._widgets[param_name] = widget
                
                # 创建标签
                label = QLabel(tr_(param.display_name) + ":")
                label.setToolTip(tr_(param.description) if param.description else "")
                
                self._params_layout.addRow(label, widget)
    
    def _create_param_widget(self, param: NodeParameter, value: Any) -> Optional[QWidget]:
        """创建参数编辑控件"""
        widget = None
        
        if param.param_type == "int":
            widget = QSpinBox()
            widget.setRange(
                param.min_value if param.min_value is not None else -999999,
                param.max_value if param.max_value is not None else 999999
            )
            widget.setValue(int(value) if value is not None else param.default_value)
            widget.valueChanged.connect(
                lambda v, pn=param.name: self._on_param_changed(pn, v)
            )
        
        elif param.param_type == "float":
            widget = QDoubleSpinBox()
            widget.setRange(
                param.min_value if param.min_value is not None else -999999.0,
                param.max_value if param.max_value is not None else 999999.0
            )
            widget.setDecimals(4)
            widget.setValue(float(value) if value is not None else param.default_value)
            widget.valueChanged.connect(
                lambda v, pn=param.name: self._on_param_changed(pn, v)
            )
        
        elif param.param_type == "str":
            widget = QLineEdit()
            widget.setText(str(value) if value is not None else str(param.default_value))
            widget.textChanged.connect(
                lambda v, pn=param.name: self._on_param_changed(pn, v)
            )
        
        elif param.param_type == "bool":
            widget = QCheckBox()
            widget.setChecked(bool(value) if value is not None else param.default_value)
            widget.stateChanged.connect(
                lambda v, pn=param.name: self._on_param_changed(pn, bool(v))
            )
        
        elif param.param_type == "choice":
            widget = QComboBox()
            if param.choices:
                for choice in param.choices:
                    widget.addItem(str(choice), choice)
                if value in param.choices:
                    widget.setCurrentIndex(param.choices.index(value))
            widget.currentIndexChanged.connect(
                lambda i, pn=param.name, w=widget: self._on_param_changed(pn, w.currentData())
            )
        
        elif param.param_type == "file":
            container = QWidget()
            layout = QHBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            
            line_edit = QLineEdit()
            line_edit.setText(str(value) if value else "")
            line_edit.textChanged.connect(
                lambda v, pn=param.name: self._on_param_changed(pn, v)
            )
            layout.addWidget(line_edit)
            
            browse_btn = QPushButton("...")
            browse_btn.setMaximumWidth(30)
            browse_btn.clicked.connect(
                lambda checked, le=line_edit, pn=param.name, flt=param.file_filter, dd=param.default_directory:
                    self._browse_file(le, pn, flt, dd)
            )
            layout.addWidget(browse_btn)
            
            widget = container
        
        elif param.param_type == "folder":
            container = QWidget()
            layout = QHBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            
            line_edit = QLineEdit()
            line_edit.setText(str(value) if value else "")
            line_edit.textChanged.connect(
                lambda v, pn=param.name: self._on_param_changed(pn, v)
            )
            layout.addWidget(line_edit)
            
            browse_btn = QPushButton("...")
            browse_btn.setMaximumWidth(30)
            browse_btn.clicked.connect(
                lambda checked, le=line_edit, pn=param.name, dd=param.default_directory:
                    self._browse_folder(le, pn, dd)
            )
            layout.addWidget(browse_btn)
            
            widget = container
        
        if widget:
            widget.setStyleSheet(Styles.FORM_CONTROLS)
        
        return widget
    
    def _on_param_changed(self, param_name: str, value: Any):
        """参数值改变回调"""
        if self._current_node and self._current_node_id:
            success, msg = self._current_node.set_parameter(param_name, value)
            if success:
                self.parameter_changed.emit(self._current_node_id, param_name, value)
            else:
                logger.warning(f"设置参数失败: {msg}")
    
    def _browse_file(self, line_edit: QLineEdit, param_name: str, file_filter: str, default_directory: str = ""):
        """浏览文件"""
        # 确定起始目录：优先使用当前值，其次使用默认目录
        current_value = line_edit.text()
        if current_value:
            start_dir = os.path.dirname(current_value) if os.path.exists(current_value) else current_value
        elif default_directory:
            start_dir = _resolve_default_directory(default_directory)
        else:
            start_dir = ""
        
        path, _ = QFileDialog.getOpenFileName(
            self,
            tr_("Select file"),
            start_dir,
            file_filter if file_filter else "All Files (*)"
        )
        if path:
            line_edit.setText(path)
    
    def _browse_folder(self, line_edit: QLineEdit, param_name: str, default_directory: str = ""):
        """浏览文件夹"""
        # 确定起始目录：优先使用当前值，其次使用默认目录
        current_value = line_edit.text()
        if current_value and os.path.exists(current_value):
            start_dir = current_value
        elif default_directory:
            start_dir = _resolve_default_directory(default_directory)
        else:
            start_dir = ""
        
        path = QFileDialog.getExistingDirectory(
            self,
            tr_("Select folder"),
            start_dir
        )
        if path:
            line_edit.setText(path)
    
    def _update_ports(self, node: BaseNode):
        """更新端口信息"""
        # 清除旧内容
        while self._ports_layout.count():
            item = self._ports_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        if not node.inputs and not node.outputs:
            self._ports_group.hide()
            return
        
        self._ports_group.show()
        
        # 输入端口
        if node.inputs:
            inputs_label = QLabel(tr_("📥 Inputs:"))
            inputs_label.setStyleSheet(f"color: {Styles.COLORS['green']}; font-weight: bold;")
            self._ports_layout.addWidget(inputs_label)
            
            for port_name, port in node.inputs.items():
                port_label = QLabel(
                    f"  • {tr_(port.display_name)} ({port.data_type.value})"
                    + (" *" if port.required else "")
                )
                port_label.setToolTip(tr_(port.description) if port.description else "")
                port_label.setStyleSheet(f"color: {Styles.COLORS['subtext1']};")
                self._ports_layout.addWidget(port_label)
        
        # 输出端口
        if node.outputs:
            outputs_label = QLabel(tr_("📤 Outputs:"))
            outputs_label.setStyleSheet(f"color: {Styles.COLORS['blue']}; font-weight: bold;")
            self._ports_layout.addWidget(outputs_label)
            
            for port_name, port in node.outputs.items():
                port_label = QLabel(
                    f"  • {tr_(port.display_name)} ({port.data_type.value})"
                )
                port_label.setToolTip(tr_(port.description) if port.description else "")
                port_label.setStyleSheet(f"color: {Styles.COLORS['subtext1']};")
                self._ports_layout.addWidget(port_label)
    
    def clear(self):
        """清除当前显示"""
        self._current_node = None
        self._current_node_id = None
        self._widgets.clear()
        self._show_empty_state()

