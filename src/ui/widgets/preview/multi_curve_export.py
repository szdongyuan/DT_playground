# -*- coding: utf-8 -*-
"""Curve selection dialog and CSV serialization for multi-curve previews."""

from __future__ import annotations

import csv
import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Sequence, Union

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from src.ui.i18n import tr_
from src.ui.styles import Styles


@dataclass(frozen=True)
class CurveExportItem:
    """A preview curve and its editable name for one CSV export."""

    label: str
    export_name: str
    x: np.ndarray
    y: np.ndarray
    is_statistic: bool = False


def _validated_arrays(
    curves: Sequence[CurveExportItem],
) -> List[tuple[CurveExportItem, np.ndarray, np.ndarray]]:
    if not curves:
        raise ValueError(tr_("Select at least one curve to export"))

    validated = []
    for curve in curves:
        x = np.asarray(curve.x)
        y = np.asarray(curve.y)
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError(
                tr_("Curve {name} must contain one-dimensional coordinates").format(
                    name=curve.label
                )
            )
        if x.size == 0 or y.size == 0:
            raise ValueError(
                tr_("Curve {name} contains no exportable points").format(
                    name=curve.label
                )
            )
        if x.size != y.size:
            raise ValueError(
                tr_("Curve {name} has mismatched coordinate and value lengths").format(
                    name=curve.label
                )
            )
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError(
                tr_("Curve {name} contains non-finite export values").format(
                    name=curve.label
                )
            )
        validated.append((curve, x, y))
    return validated


def _write_rows(file, curves: Sequence[CurveExportItem]) -> None:
    validated = _validated_arrays(curves)
    first_x = validated[0][1]
    shared_x = all(
        x.shape == first_x.shape and np.array_equal(x, first_x)
        for _, x, _ in validated[1:]
    )
    writer = csv.writer(file)

    if shared_x:
        writer.writerow(["x", *(curve.export_name for curve, _, _ in validated)])
        for point_index in range(first_x.size):
            writer.writerow(
                [
                    str(first_x[point_index]),
                    *(str(y[point_index]) for _, _, y in validated),
                ]
            )
        return

    header = []
    for curve, _, _ in validated:
        header.extend([f"{curve.export_name}_x", f"{curve.export_name}_y"])
    writer.writerow(header)

    row_count = max(x.size for _, x, _ in validated)
    for point_index in range(row_count):
        row = []
        for _, x, y in validated:
            if point_index < x.size:
                row.extend([str(x[point_index]), str(y[point_index])])
            else:
                row.extend(["", ""])
        writer.writerow(row)


def write_curve_csv(
    path: Union[str, os.PathLike[str]], curves: Sequence[CurveExportItem]
) -> None:
    """Write selected preview curves atomically as a UTF-8 BOM CSV file."""

    _validated_arrays(curves)
    target = Path(path)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8-sig",
            newline="",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            _write_rows(temporary_file, curves)
        os.replace(temporary_path, target)
    except Exception:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


class MultiCurveExportDialog(QDialog):
    """Select and rename currently visible curves for a single export."""

    def __init__(
        self,
        candidates: Sequence[CurveExportItem],
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        statistics = [item for item in candidates if item.is_statistic]
        raw_curves = [item for item in candidates if not item.is_statistic]
        self._ordered_candidates = [*statistics, *raw_curves]
        self._row_checkboxes: List[QCheckBox] = []
        self._name_edits: List[QLineEdit] = []
        self._updating_selection = False
        self._setup_ui(statistics, raw_curves)

    def _setup_ui(
        self,
        statistics: Sequence[CurveExportItem],
        raw_curves: Sequence[CurveExportItem],
    ) -> None:
        self.setWindowTitle(tr_("Export curves to CSV"))
        self.setModal(True)
        self.resize(620, 520)
        self.setStyleSheet(
            Styles.CHECKBOX_CONTROLS
            + f"""
            QDialog {{
                background: {Styles.COLORS['base']};
                color: {Styles.COLORS['text']};
            }}
            QLabel, QCheckBox {{ color: {Styles.COLORS['text']}; }}
            QLineEdit {{
                background: {Styles.COLORS['surface0']};
                color: {Styles.COLORS['text']};
                border: 1px solid {Styles.COLORS['surface1']};
                border-radius: 4px;
                padding: 5px;
            }}
            QPushButton {{
                background: {Styles.COLORS['surface1']};
                color: {Styles.COLORS['text']};
                border: none;
                border-radius: 4px;
                padding: 6px 14px;
            }}
            QPushButton:disabled {{ color: {Styles.COLORS['overlay0']}; }}
            QScrollArea {{ border: 1px solid {Styles.COLORS['surface1']}; }}
            """
        )

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(tr_("Select visible curves and edit their CSV column names."))
        )

        self._select_all_checkbox = QCheckBox(tr_("Select all"))
        self._select_all_checkbox.setTristate(True)
        self._select_all_checkbox.clicked.connect(self._on_select_all_clicked)
        layout.addWidget(self._select_all_checkbox)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_content.setStyleSheet(
            f"background: {Styles.COLORS['mantle']}; color: {Styles.COLORS['text']};"
        )
        rows_layout = QVBoxLayout(scroll_content)
        rows_layout.setContentsMargins(10, 10, 10, 10)
        rows_layout.setSpacing(7)

        if statistics:
            rows_layout.addWidget(self._section_label(tr_("Summary curves")))
            for candidate in statistics:
                self._add_candidate_row(rows_layout, candidate)

        if statistics and raw_curves:
            self._separator = QFrame()
            self._separator.setFrameShape(QFrame.Shape.HLine)
            self._separator.setFrameShadow(QFrame.Shadow.Sunken)
            self._separator.setStyleSheet(
                f"color: {Styles.COLORS['surface1']}; margin: 6px 0;"
            )
            rows_layout.addWidget(self._separator)
        else:
            self._separator = None

        if raw_curves:
            rows_layout.addWidget(self._section_label(tr_("Original curves")))
            for candidate in raw_curves:
                self._add_candidate_row(rows_layout, candidate)

        rows_layout.addStretch()
        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area, 1)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        self._confirm_button = button_box.button(
            QDialogButtonBox.StandardButton.Ok
        )
        self._confirm_button.setText(tr_("Export"))
        self._confirm_button.setEnabled(False)
        button_box.button(QDialogButtonBox.StandardButton.Cancel).setText(
            tr_("Cancel")
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    @staticmethod
    def _section_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(
            f"color: {Styles.COLORS['blue']}; font-weight: bold; padding-top: 2px;"
        )
        return label

    def _add_candidate_row(
        self, layout: QVBoxLayout, candidate: CurveExportItem
    ) -> None:
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)

        checkbox = QCheckBox(candidate.label)
        checkbox.setMinimumWidth(210)
        checkbox.stateChanged.connect(self._update_selection_state)
        name_edit = QLineEdit(candidate.export_name)
        name_edit.setPlaceholderText(tr_("CSV column name"))

        row_layout.addWidget(checkbox, 1)
        row_layout.addWidget(name_edit, 1)
        layout.addWidget(row_widget)
        self._row_checkboxes.append(checkbox)
        self._name_edits.append(name_edit)

    def _on_select_all_clicked(self, checked: bool) -> None:
        self._updating_selection = True
        try:
            for checkbox in self._row_checkboxes:
                checkbox.setChecked(checked)
        finally:
            self._updating_selection = False
        self._update_selection_state()

    def _update_selection_state(self, _state: int | None = None) -> None:
        if self._updating_selection:
            return
        selected_count = sum(
            checkbox.isChecked() for checkbox in self._row_checkboxes
        )
        if selected_count == 0:
            master_state = Qt.CheckState.Unchecked
        elif selected_count == len(self._row_checkboxes):
            master_state = Qt.CheckState.Checked
        else:
            master_state = Qt.CheckState.PartiallyChecked

        self._select_all_checkbox.blockSignals(True)
        self._select_all_checkbox.setCheckState(master_state)
        self._select_all_checkbox.blockSignals(False)
        self._confirm_button.setEnabled(selected_count > 0)

    def selected_curves(self) -> List[CurveExportItem]:
        """Return checked curves with the names currently entered by the user."""

        return [
            replace(candidate, export_name=name_edit.text())
            for candidate, checkbox, name_edit in zip(
                self._ordered_candidates,
                self._row_checkboxes,
                self._name_edits,
            )
            if checkbox.isChecked()
        ]


__all__ = [
    "CurveExportItem",
    "MultiCurveExportDialog",
    "write_curve_csv",
]
