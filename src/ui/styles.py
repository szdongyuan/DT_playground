"""
Unified Style Management Module
Centrally manages all UI component styles to avoid code duplication
"""


class Styles:
    """统一样式类"""
    
    # ===== 颜色定义 (Catppuccin Mocha) =====
    COLORS = {
        # 基础色
        'base': '#1e1e2e',
        'mantle': '#181825',
        'crust': '#11111b',
        # 表面色
        'surface0': '#313244',
        'surface1': '#45475a',
        'surface2': '#585b70',
        # 覆盖色
        'overlay0': '#6c7086',
        'overlay1': '#7f849c',
        'overlay2': '#9399b2',
        # 文字色
        'text': '#cdd6f4',
        'subtext': '#a6adc8',
        'subtext0': '#a6adc8',
        'subtext1': '#bac2de',
        # 主题色
        'blue': '#89b4fa',
        'lavender': '#b4befe',
        'sapphire': '#74c7ec',
        'sky': '#89dceb',
        'teal': '#94e2d5',
        'green': '#a6e3a1',
        'yellow': '#f9e2af',
        'peach': '#fab387',
        'maroon': '#eba0ac',
        'red': '#f38ba8',
        'mauve': '#cba6f7',
        'pink': '#f5c2e7',
        'flamingo': '#f2cdcd',
        'rosewater': '#f5e0dc',
        # 别名
        'purple': '#cba6f7',
    }
    
    # ===== 节点执行状态颜色 =====
    NODE_STATE_COLORS = {
        'idle': None,                # 空闲 - 无额外装饰
        'running': '#f9e2af',        # 运行中 - 黄色
        'completed': '#a6e3a1',      # 已完成 - 绿色
        'error': '#f38ba8',          # 错误 - 红色
        'waiting': '#89b4fa',        # 等待中 - 蓝色
    }
    
    # ===== 节点状态图标 =====
    NODE_STATE_ICONS = {
        'idle': '',
        'running': '⏳',
        'completed': '✅',
        'error': '❌',
        'waiting': '⏸️',
    }
    
    # ===== Checkbox controls =====
    CHECKBOX_CONTROLS = """
        QCheckBox {
            color: #cdd6f4;
            spacing: 8px;
        }
        QCheckBox::indicator {
            width: 16px;
            height: 16px;
            border: 2px solid #7f849c;
            border-radius: 3px;
            background-color: #313244;
        }
        QCheckBox::indicator:hover {
            border-color: #89b4fa;
        }
        QCheckBox::indicator:checked {
            background-color: #89b4fa;
            border-color: #b4befe;
        }
        QCheckBox::indicator:indeterminate {
            background-color: #89b4fa;
            border-color: #f9e2af;
        }
        QCheckBox::indicator:disabled {
            background-color: #313244;
            border-color: #45475a;
        }
    """

    # ===== 通用表单控件样式 =====
    FORM_CONTROLS = CHECKBOX_CONTROLS + """
        QLabel {
            color: #cdd6f4;
        }
        QSpinBox, QDoubleSpinBox {
            background-color: #313244;
            border: 1px solid #45475a;
            border-radius: 4px;
            padding: 4px;
            color: #cdd6f4;
        }
        QComboBox {
            background-color: #313244;
            border: 1px solid #45475a;
            border-radius: 4px;
            padding: 4px;
            color: #cdd6f4;
        }
        QComboBox::drop-down {
            border: none;
        }
        QComboBox::down-arrow {
            image: none;
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-top: 6px solid #cdd6f4;
        }
        QComboBox QAbstractItemView {
            background-color: #313244;
            color: #cdd6f4;
            selection-background-color: #45475a;
        }
        QPushButton {
            background-color: #45475a;
            border: none;
            padding: 6px 12px;
            border-radius: 4px;
            color: #cdd6f4;
        }
        QPushButton:hover {
            background-color: #585b70;
        }
    """
    
    # ===== GroupBox 样式 =====
    @staticmethod
    def group_box(color: str) -> str:
        """获取 GroupBox 样式"""
        return f"""
            QGroupBox {{
                font-weight: bold;
                border: 1px solid #45475a;
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 10px;
                color: #cdd6f4;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: {color};
            }}
            QComboBox {{
                background-color: #313244;
                border: 1px solid #45475a;
                border-radius: 4px;
                padding: 4px;
                color: #cdd6f4;
            }}
            QComboBox::drop-down {{
                border: none;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid #cdd6f4;
            }}
            QComboBox QAbstractItemView {{
                background-color: #313244;
                color: #cdd6f4;
                selection-background-color: #45475a;
            }}
            QPushButton {{
                background-color: #45475a;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                color: #cdd6f4;
            }}
            QPushButton:hover {{
                background-color: #585b70;
            }}
        """

    @staticmethod
    def canvas_tab_widget() -> str:
        """Return the shared dark IDE-style tab strip used above graph canvases."""
        return f"""
            QTabWidget {{
                background: {Styles.COLORS['base']};
            }}
            QTabWidget::pane {{
                border: 1px solid {Styles.COLORS['surface1']};
                border-top: 1px solid {Styles.COLORS['surface1']};
                background: {Styles.COLORS['base']};
            }}
            QTabWidget::tab-bar {{
                alignment: left;
            }}
            QTabBar {{
                background: {Styles.COLORS['mantle']};
                border-top: 1px solid rgba(88, 91, 112, 0.36);
                border-bottom: 1px solid {Styles.COLORS['surface1']};
            }}
            QTabBar::tab {{
                background: rgba(49, 50, 68, 0.72);
                color: {Styles.COLORS['subtext1']};
                border: 1px solid {Styles.COLORS['surface1']};
                border-bottom: 2px solid transparent;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                padding: 7px 14px 6px 14px;
                margin-right: 2px;
                min-width: 110px;
            }}
            QTabBar::tab:selected {{
                background: {Styles.COLORS['surface2']};
                color: {Styles.COLORS['text']};
                border-color: {Styles.COLORS['blue']};
                border-bottom: 2px solid {Styles.COLORS['blue']};
            }}
            QTabBar::tab:!selected {{
                background: rgba(49, 50, 68, 0.62);
                color: {Styles.COLORS['subtext1']};
            }}
            QTabBar::tab:hover {{
                background: {Styles.COLORS['surface1']};
                color: {Styles.COLORS['text']};
                border-bottom: 2px solid {Styles.COLORS['lavender']};
            }}
            QToolButton {{
                background: {Styles.COLORS['surface0']};
                color: {Styles.COLORS['text']};
                border: 1px solid {Styles.COLORS['surface1']};
                border-radius: 5px;
                padding: 4px 10px;
                margin: 3px;
                font-weight: 700;
            }}
            QToolButton:hover {{
                background: {Styles.COLORS['surface1']};
                border-color: {Styles.COLORS['blue']};
            }}
        """
    
    # ===== 按钮样式 =====
    BUTTON_PRIMARY = """
        QPushButton {
            background-color: #a6e3a1;
            color: #1e1e2e;
            font-weight: bold;
            padding: 12px;
            font-size: 14px;
        }
        QPushButton:hover {
            background-color: #94e2d5;
        }
        QPushButton:disabled {
            background-color: #45475a;
            color: #6c7086;
        }
    """
    
    BUTTON_DANGER = """
        QPushButton {
            background-color: #f38ba8;
            color: #1e1e2e;
            font-weight: bold;
            padding: 12px;
            font-size: 14px;
        }
        QPushButton:hover {
            background-color: #eba0ac;
        }
        QPushButton:disabled {
            background-color: #45475a;
            color: #6c7086;
        }
    """
    
    BUTTON_ACCENT = """
        QPushButton {
            background-color: #cba6f7;
            color: #1e1e2e;
            font-weight: bold;
            padding: 10px;
        }
        QPushButton:hover {
            background-color: #f5c2e7;
        }
    """
    
    # ===== 进度条样式 =====
    PROGRESS_BAR = """
        QProgressBar {
            border: 1px solid #45475a;
            border-radius: 4px;
            text-align: center;
            background-color: #313244;
            height: 20px;
        }
        QProgressBar::chunk {
            background-color: #89b4fa;
            border-radius: 3px;
        }
    """
    
    # ===== 树形控件样式 =====
    TREE_WIDGET = """
        QTreeWidget {
            background-color: #181825;
            border: 1px solid #45475a;
            border-radius: 4px;
            color: #cdd6f4;
        }
        QTreeWidget::item {
            padding: 4px;
            color: #cdd6f4;
        }
        QTreeWidget::item:selected {
            background-color: #45475a;
            color: #cdd6f4;
        }
        QTreeWidget::item:hover {
            background-color: #313244;
        }
        QHeaderView::section {
            background-color: #313244;
            color: #cdd6f4;
            padding: 4px;
            border: none;
        }
    """
    
    # ===== 文本编辑框样式 =====
    TEXT_EDIT = """
        QTextEdit {
            background-color: #181825;
            border: 1px solid #45475a;
            border-radius: 4px;
            font-family: Consolas, monospace;
            font-size: 11px;
            color: #cdd6f4;
        }
    """
    
    # ===== 滚动区域样式 =====
    SCROLL_AREA = """
        QScrollArea {
            border: none;
            background-color: transparent;
        }
    """
    
    # ===== 指标显示框样式 =====
    METRIC_FRAME = """
        background-color: #313244;
        border-radius: 4px;
        padding: 8px;
    """
    
    METRIC_TITLE = "color: #a6adc8; font-size: 11px;"
    
    @staticmethod
    def metric_value(color: str) -> str:
        """获取指标值样式"""
        return f"color: {color}; font-size: 18px; font-weight: bold;"
    
    @staticmethod
    def get_color(color_str: str):
        """获取 QColor 对象"""
        from PySide6.QtGui import QColor
        
        # 如果是颜色名称，从 COLORS 字典获取
        if color_str in Styles.COLORS:
            return QColor(Styles.COLORS[color_str])
        
        # 否则直接解析颜色字符串
        return QColor(color_str)

