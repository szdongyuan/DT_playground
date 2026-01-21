"""
Dataset Dialog
"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
    QFormLayout, QGroupBox, QHBoxLayout, QHeaderView,
    QLabel, QMessageBox, QPushButton, QSpinBox,
    QTableWidget, QTableWidgetItem, QVBoxLayout
)


class DatasetSplitDialog(QDialog):
    """数据集划分对话框"""
    
    split_confirmed = pyqtSignal(dict)  # 划分确认
    
    def __init__(self, parent=None, total_samples: int = 0, labels: list = None):
        super().__init__(parent)
        self.total_samples = total_samples
        self.labels = labels or []
        self._init_ui()
    
    def _init_ui(self):
        """初始化界面"""
        self.setWindowTitle("数据集划分")
        self.setMinimumSize(500, 450)
        self.setModal(True)
        
        layout = QVBoxLayout(self)
        
        # 数据集信息
        info_group = QGroupBox("数据集信息")
        info_layout = QFormLayout(info_group)
        
        self.total_label = QLabel(str(self.total_samples))
        self.total_label.setStyleSheet("font-weight: bold; color: #89b4fa;")
        info_layout.addRow("总样本数:", self.total_label)
        
        self.classes_label = QLabel(str(len(self.labels)))
        self.classes_label.setStyleSheet("font-weight: bold; color: #a6e3a1;")
        info_layout.addRow("类别数:", self.classes_label)
        
        # 显示类别名称
        if self.labels:
            self.class_names_label = QLabel(", ".join(self.labels))
            self.class_names_label.setWordWrap(True)
            self.class_names_label.setStyleSheet("color: #cba6f7;")
            info_layout.addRow("类别:", self.class_names_label)
        
        layout.addWidget(info_group)
        
        # 划分比例
        split_group = QGroupBox("划分比例")
        split_layout = QFormLayout(split_group)
        
        # 训练集
        train_layout = QHBoxLayout()
        self.train_spin = QSpinBox()
        self.train_spin.setRange(10, 90)
        self.train_spin.setValue(70)
        self.train_spin.setSuffix(" %")
        self.train_spin.valueChanged.connect(self._update_splits)
        train_layout.addWidget(self.train_spin)
        
        self.train_count = QLabel("0 样本")
        self.train_count.setStyleSheet("color: #a6adc8;")
        train_layout.addWidget(self.train_count)
        train_layout.addStretch()
        
        split_layout.addRow("训练集:", train_layout)
        
        # 验证集
        val_layout = QHBoxLayout()
        self.val_spin = QSpinBox()
        self.val_spin.setRange(5, 40)
        self.val_spin.setValue(15)
        self.val_spin.setSuffix(" %")
        self.val_spin.valueChanged.connect(self._update_splits)
        val_layout.addWidget(self.val_spin)
        
        self.val_count = QLabel("0 样本")
        self.val_count.setStyleSheet("color: #a6adc8;")
        val_layout.addWidget(self.val_count)
        val_layout.addStretch()
        
        split_layout.addRow("验证集:", val_layout)
        
        # 测试集
        test_layout = QHBoxLayout()
        self.test_spin = QSpinBox()
        self.test_spin.setRange(5, 40)
        self.test_spin.setValue(15)
        self.test_spin.setSuffix(" %")
        self.test_spin.valueChanged.connect(self._update_splits)
        test_layout.addWidget(self.test_spin)
        
        self.test_count = QLabel("0 样本")
        self.test_count.setStyleSheet("color: #a6adc8;")
        test_layout.addWidget(self.test_count)
        test_layout.addStretch()
        
        split_layout.addRow("测试集:", test_layout)
        
        # 总和提示
        self.sum_label = QLabel("总计: 100%")
        self.sum_label.setStyleSheet("font-weight: bold;")
        split_layout.addRow("", self.sum_label)
        
        layout.addWidget(split_group)
        
        # 划分选项
        options_group = QGroupBox("划分选项")
        options_layout = QVBoxLayout(options_group)
        
        self.stratify_check = QCheckBox("分层抽样 (保持类别比例)")
        self.stratify_check.setChecked(True)
        options_layout.addWidget(self.stratify_check)
        
        self.shuffle_check = QCheckBox("打乱数据")
        self.shuffle_check.setChecked(True)
        options_layout.addWidget(self.shuffle_check)
        
        seed_layout = QHBoxLayout()
        seed_layout.addWidget(QLabel("随机种子:"))
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 99999)
        self.seed_spin.setValue(42)
        seed_layout.addWidget(self.seed_spin)
        seed_layout.addStretch()
        options_layout.addLayout(seed_layout)
        
        layout.addWidget(options_group)
        
        # 类别分布表
        if self.labels:
            dist_group = QGroupBox("类别分布")
            dist_layout = QVBoxLayout(dist_group)
            
            self.dist_table = QTableWidget()
            self.dist_table.setColumnCount(4)
            self.dist_table.setHorizontalHeaderLabels(["类别", "总数", "训练", "验证/测试"])
            self.dist_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            self.dist_table.setMaximumHeight(150)
            
            dist_layout.addWidget(self.dist_table)
            layout.addWidget(dist_group)
        
        layout.addStretch()
        
        # 按钮
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.cancel_btn)
        
        self.confirm_btn = QPushButton("确认划分")
        self.confirm_btn.setStyleSheet("""
            QPushButton {
                background-color: #a6e3a1;
                color: #1e1e2e;
                font-weight: bold;
            }
        """)
        self.confirm_btn.clicked.connect(self._confirm_split)
        btn_layout.addWidget(self.confirm_btn)
        
        layout.addLayout(btn_layout)
        
        self._apply_styles()
        self._update_splits()
    
    def _update_splits(self):
        """更新划分统计"""
        train_pct = self.train_spin.value()
        val_pct = self.val_spin.value()
        test_pct = self.test_spin.value()
        
        total_pct = train_pct + val_pct + test_pct
        
        # 更新样本数
        train_count = int(self.total_samples * train_pct / 100)
        val_count = int(self.total_samples * val_pct / 100)
        test_count = int(self.total_samples * test_pct / 100)
        
        self.train_count.setText(f"{train_count} 样本")
        self.val_count.setText(f"{val_count} 样本")
        self.test_count.setText(f"{test_count} 样本")
        
        # 更新总和
        if total_pct == 100:
            self.sum_label.setText("总计: 100% ✓")
            self.sum_label.setStyleSheet("font-weight: bold; color: #a6e3a1;")
            self.confirm_btn.setEnabled(True)
        else:
            self.sum_label.setText(f"总计: {total_pct}% (需要100%)")
            self.sum_label.setStyleSheet("font-weight: bold; color: #f38ba8;")
            self.confirm_btn.setEnabled(False)
    
    def _confirm_split(self):
        """确认划分"""
        train_pct = self.train_spin.value()
        val_pct = self.val_spin.value()
        test_pct = self.test_spin.value()
        
        if train_pct + val_pct + test_pct != 100:
            QMessageBox.warning(self, "警告", "划分比例之和必须为100%")
            return
        
        config = {
            'train_ratio': train_pct / 100,
            'val_ratio': val_pct / 100,
            'test_ratio': test_pct / 100,
            'stratify': self.stratify_check.isChecked(),
            'shuffle': self.shuffle_check.isChecked(),
            'random_seed': self.seed_spin.value()
        }
        
        self.split_confirmed.emit(config)
        self.accept()
    
    def set_total_samples(self, count: int):
        """设置总样本数"""
        self.total_samples = count
        self.total_label.setText(str(count))
        self._update_splits()
    
    def set_labels(self, labels: list):
        """设置标签列表"""
        self.labels = labels
        self.classes_label.setText(str(len(labels)))
    
    def _apply_styles(self):
        """应用样式"""
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e2e;
                color: #cdd6f4;
            }
            QLabel {
                color: #cdd6f4;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #45475a;
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 10px;
                color: #cdd6f4;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #89b4fa;
            }
            QSpinBox, QDoubleSpinBox {
                background-color: #313244;
                border: 1px solid #45475a;
                border-radius: 4px;
                padding: 4px;
                min-width: 80px;
                color: #cdd6f4;
            }
            QCheckBox {
                color: #cdd6f4;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #45475a;
                border-radius: 3px;
                background-color: #313244;
            }
            QCheckBox::indicator:checked {
                background-color: #89b4fa;
                border-color: #89b4fa;
            }
            QPushButton {
                background-color: #45475a;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                color: #cdd6f4;
            }
            QPushButton:hover {
                background-color: #585b70;
            }
            QTableWidget {
                background-color: #181825;
                border: 1px solid #45475a;
                gridline-color: #313244;
                color: #cdd6f4;
            }
            QTableWidget::item {
                color: #cdd6f4;
            }
            QHeaderView::section {
                background-color: #313244;
                padding: 4px;
                border: none;
                color: #cdd6f4;
            }
        """)

