"""Preview a structured classification evaluation result."""

import logging
from typing import Any, Dict, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHeaderView,
    QLabel,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.ui.i18n import tr_
from src.ui.styles import Styles

from .base_preview import BasePreviewWidget, register_preview


logger = logging.getLogger(__name__)


@register_preview
class ClassificationResultPreviewWidget(BasePreviewWidget):
    """Show summary, per-class, confusion, and prediction views."""

    display_name = tr_("Classification result")
    icon = "📊"
    MAX_PREDICTION_ROWS = 1000

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        summary_group = QGroupBox(tr_("Summary metrics"))
        summary_group.setStyleSheet(Styles.group_box(Styles.COLORS["blue"]))
        summary_layout = QVBoxLayout(summary_group)
        self._summary_label = QLabel()
        self._summary_label.setWordWrap(True)
        self._summary_label.setTextFormat(Qt.TextFormat.RichText)
        summary_layout.addWidget(self._summary_label)
        self._warnings_label = QLabel()
        self._warnings_label.setWordWrap(True)
        self._warnings_label.setStyleSheet(
            f"color: {Styles.COLORS['yellow']}; padding-top: 4px;"
        )
        summary_layout.addWidget(self._warnings_label)
        layout.addWidget(summary_group)

        self._tabs = QTabWidget()
        self._per_class_table = self._table(
            [
                tr_("Class ID"),
                tr_("Class name"),
                tr_("Precision"),
                tr_("Recall"),
                tr_("F1"),
                tr_("Support"),
            ]
        )
        self._confusion_table = QTableWidget()
        self._confusion_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._confusion_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._confusion_table.verticalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._predictions_table = self._table(
            [
                tr_("Index"),
                tr_("File path"),
                tr_("True class"),
                tr_("Predicted class"),
                tr_("Confidence"),
            ]
        )
        self._only_errors = QCheckBox(tr_("Only misclassified"))
        self._only_errors.toggled.connect(self._refresh_predictions)
        self._tabs.addTab(self._wrap(self._per_class_table), tr_("Per-class metrics"))
        self._tabs.addTab(self._wrap(self._confusion_table), tr_("Confusion matrix"))
        predictions_container = QWidget()
        predictions_layout = QVBoxLayout(predictions_container)
        predictions_layout.setContentsMargins(0, 0, 0, 0)
        predictions_layout.addWidget(self._only_errors)
        predictions_layout.addWidget(self._predictions_table)
        self._tabs.addTab(predictions_container, tr_("Predictions"))
        layout.addWidget(self._tabs, 1)

    @staticmethod
    def _wrap(widget: QWidget) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(widget)
        return container

    @staticmethod
    def _table(headers) -> QTableWidget:
        table = QTableWidget()
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setAlternatingRowColors(True)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        return table

    def set_data(self, data: Any) -> bool:
        self._current_data = data
        if not self.can_display(data):
            self.clear()
            return False

        try:
            classes = data["classes"]
            self._update_data_info(
                "CLASSIFICATION_RESULT "
                f"samples={data.get('sample_count', 0)} classes={len(classes)}"
            )
            self._update_summary(data["metrics"], data.get("warnings", []), classes)
            self._update_per_class(data["per_class"])
            self._update_confusion(data["confusion_matrix"], classes)
            self._update_predictions(data["predictions"])
            self.data_changed.emit()
            return True
        except Exception as exc:
            logger.error("Failed to display classification result: %s", exc)
            self.clear()
            return False

    def _update_summary(self, metrics: Dict, warnings, classes):
        preferred = [
            "loss",
            "accuracy",
            "balanced_accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "weighted_precision",
            "weighted_recall",
            "weighted_f1",
        ]
        lines = []
        for name in preferred:
            if name not in metrics:
                continue
            value = float(metrics[name])
            suffix = f" ({value * 100:.2f}%)" if name != "loss" else ""
            lines.append(f"<b>{name}</b>: {value:.6f}{suffix}")
        self._summary_label.setText("<br>".join(lines))

        class_names = {item["id"]: item["name"] for item in classes}
        warning_lines = []
        for warning in warnings:
            names = ", ".join(
                class_names.get(class_id, str(class_id))
                for class_id in warning.get("class_ids", [])
            )
            if warning.get("code") == "missing_true_classes":
                warning_lines.append(
                    tr_("Classes absent from test targets: {classes}").format(classes=names)
                )
            elif warning.get("code") == "unpredicted_classes":
                warning_lines.append(
                    tr_("Classes never predicted: {classes}").format(classes=names)
                )
        self._warnings_label.setText("\n".join(warning_lines))
        self._warnings_label.setVisible(bool(warning_lines))

    def _update_per_class(self, rows):
        self._per_class_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            values = [
                row["class_id"],
                row["class_name"],
                f"{float(row['precision']):.6f}",
                f"{float(row['recall']):.6f}",
                f"{float(row['f1']):.6f}",
                row["support"],
            ]
            for column, value in enumerate(values):
                self._per_class_table.setItem(
                    row_index, column, QTableWidgetItem(str(value))
                )

    def _update_confusion(self, matrix, classes):
        size = len(classes)
        names = [str(item["name"]) for item in classes]
        self._confusion_table.setRowCount(size)
        self._confusion_table.setColumnCount(size)
        self._confusion_table.setHorizontalHeaderLabels(names)
        self._confusion_table.setVerticalHeaderLabels(names)
        for row_index, row in enumerate(matrix):
            for column, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._confusion_table.setItem(row_index, column, item)

    def _update_predictions(self, predictions):
        self._prediction_rows = list(predictions)
        self._refresh_predictions()

    def _refresh_predictions(self):
        predictions = getattr(self, "_prediction_rows", [])
        if self._only_errors.isChecked():
            predictions = [
                row
                for row in predictions
                if row.get("true_class_id") != row.get("predicted_class_id")
            ]
        display_rows = predictions[: self.MAX_PREDICTION_ROWS]
        self._predictions_table.setRowCount(len(display_rows))
        for row_index, row in enumerate(display_rows):
            values = [
                row["index"],
                row.get("file_path") or "",
                row["true_class_name"],
                row["predicted_class_name"],
                f"{float(row['confidence']):.6f}",
            ]
            for column, value in enumerate(values):
                self._predictions_table.setItem(
                    row_index, column, QTableWidgetItem(str(value))
                )

    def clear(self):
        self._current_data = None
        self._data_info = ""
        self._summary_label.setText("")
        self._warnings_label.setText("")
        self._warnings_label.hide()
        self._prediction_rows = []
        self._only_errors.setChecked(False)
        self._per_class_table.setRowCount(0)
        self._confusion_table.setRowCount(0)
        self._confusion_table.setColumnCount(0)
        self._predictions_table.setRowCount(0)

    @classmethod
    def can_display(cls, data: Any) -> bool:
        return (
            isinstance(data, dict)
            and data.get("task") == "classification"
            and isinstance(data.get("metrics"), dict)
            and isinstance(data.get("classes"), list)
            and isinstance(data.get("confusion_matrix"), list)
            and isinstance(data.get("per_class"), list)
            and isinstance(data.get("predictions"), list)
        )
