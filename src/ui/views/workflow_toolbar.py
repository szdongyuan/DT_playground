# -*- coding: utf-8 -*-
"""
Workflow Toolbar

Reusable toolbar component for workflow views.
Provides title label, command buttons and breakpoint mode toggle.
Communicates purely via signals -- contains no workflow / editor logic.
"""

from typing import Optional

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QWidget,
)

from ..i18n import tr_
from ..styles import Styles


class WorkflowToolbar(QFrame):
    """
    Reusable workflow toolbar.

    Signals are emitted when the user clicks the corresponding button.
    The container / parent is responsible for connecting them to actual logic.
    """

    new_requested = pyqtSignal()
    open_requested = pyqtSignal()
    save_requested = pyqtSignal()
    duplicate_requested = pyqtSignal()
    run_requested = pyqtSignal()
    stop_requested = pyqtSignal()
    continue_requested = pyqtSignal()
    fit_requested = pyqtSignal()
    clear_requested = pyqtSignal()

    def __init__(self, parent=None, *, show_duplicate: bool = False):
        super().__init__(parent)
        self._show_duplicate = show_duplicate
        self._setup_ui()

    # ===== Public API =====

    def set_title(self, text: str):
        self._title_label.setText(f"🔧 {text}")

    def set_run_controls_state(
        self,
        *,
        run_enabled: bool,
        stop_enabled: bool,
        stop_text: Optional[str] = None,
    ) -> None:
        self._run_btn.setEnabled(run_enabled)
        self._stop_btn.setEnabled(stop_enabled)
        if stop_text is not None:
            self._stop_btn.setText(stop_text)

    def show_breakpoint_mode(self, enabled: bool, node_name: str = ""):
        self._run_control_stack.setCurrentIndex(1 if enabled else 0)
        if enabled and node_name:
            self._breakpoint_label.setText(f"🔴 {node_name}")
        else:
            self._breakpoint_label.setText(tr_("🔴 Paused"))

    # ===== UI construction =====

    def _setup_ui(self):
        self.setFixedHeight(42)
        self.setStyleSheet(f"""
            QFrame {{
                background: {Styles.COLORS['surface0']};
                border-bottom: 1px solid {Styles.COLORS['surface1']};
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 4, 12, 4)
        layout.setSpacing(8)

        self._title_label = QLabel("🔧 new_workflow")
        self._title_label.setStyleSheet(f"""
            font-size: 14px;
            font-weight: bold;
            color: {Styles.COLORS['text']};
        """)
        layout.addWidget(self._title_label)

        layout.addStretch()

        btn_style = f"""
            QPushButton {{
                background: {Styles.COLORS['surface1']};
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                color: {Styles.COLORS['text']};
            }}
            QPushButton:hover {{
                background: {Styles.COLORS['surface2']};
            }}
        """

        new_btn = QPushButton(tr_("📄 New"))
        new_btn.setStyleSheet(btn_style)
        new_btn.clicked.connect(self.new_requested)
        layout.addWidget(new_btn)

        open_btn = QPushButton(tr_("📂 Open"))
        open_btn.setStyleSheet(btn_style)
        open_btn.clicked.connect(self.open_requested)
        layout.addWidget(open_btn)

        save_btn = QPushButton(tr_("💾 Save"))
        save_btn.setStyleSheet(btn_style)
        save_btn.clicked.connect(self.save_requested)
        layout.addWidget(save_btn)

        if self._show_duplicate:
            dup_btn = QPushButton(tr_("📑 Duplicate"))
            dup_btn.setStyleSheet(btn_style)
            dup_btn.clicked.connect(self.duplicate_requested)
            layout.addWidget(dup_btn)

        # Run / Stop / Breakpoint controls via QStackedWidget
        self._run_control_stack = QStackedWidget()
        self._run_control_stack.setFixedHeight(32)
        self._run_control_stack.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Fixed,
        )
        self._run_control_stack.setStyleSheet("QStackedWidget { background: transparent; }")

        # Page 0: normal mode (Run + Stop)
        normal_widget = QWidget()
        normal_widget.setStyleSheet("QWidget { background: transparent; }")
        normal_layout = QHBoxLayout(normal_widget)
        normal_layout.setContentsMargins(0, 0, 0, 0)
        normal_layout.setSpacing(4)

        self._run_btn = QPushButton(tr_("▶️ Run"))
        self._run_btn.setStyleSheet(btn_style)
        self._run_btn.clicked.connect(self.run_requested)
        normal_layout.addWidget(self._run_btn)

        self._stop_btn = QPushButton(tr_("⏹️ Stop"))
        self._stop_btn.setStyleSheet(btn_style)
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self.stop_requested)
        normal_layout.addWidget(self._stop_btn)

        self._run_control_stack.addWidget(normal_widget)  # index 0

        # Page 1: breakpoint mode (label + Continue)
        breakpoint_widget = QWidget()
        breakpoint_widget.setStyleSheet("QWidget { background: transparent; }")
        breakpoint_layout = QHBoxLayout(breakpoint_widget)
        breakpoint_layout.setContentsMargins(0, 0, 0, 0)
        breakpoint_layout.setSpacing(4)

        self._breakpoint_label = QLabel(tr_("🔴 Paused"))
        self._breakpoint_label.setStyleSheet(f"""
            QLabel {{
                color: {Styles.COLORS['red']};
                font-weight: bold;
                padding: 6px 12px;
                background: {Styles.COLORS['surface1']};
                border-radius: 4px;
            }}
        """)
        breakpoint_layout.addWidget(self._breakpoint_label)

        continue_btn = QPushButton(tr_("⏵ Continue"))
        continue_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Styles.COLORS['green']};
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                color: {Styles.COLORS['crust']};
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: {Styles.COLORS['teal']};
            }}
        """)
        continue_btn.clicked.connect(self.continue_requested)
        breakpoint_layout.addWidget(continue_btn)

        self._run_control_stack.addWidget(breakpoint_widget)  # index 1
        self._run_control_stack.setCurrentIndex(0)
        layout.addWidget(self._run_control_stack)

        fit_btn = QPushButton(tr_("🔍 Fit"))
        fit_btn.setStyleSheet(btn_style)
        fit_btn.clicked.connect(self.fit_requested)
        layout.addWidget(fit_btn)

        clear_btn = QPushButton(tr_("🗑️ Clear"))
        clear_btn.setStyleSheet(btn_style)
        clear_btn.clicked.connect(self.clear_requested)
        layout.addWidget(clear_btn)
