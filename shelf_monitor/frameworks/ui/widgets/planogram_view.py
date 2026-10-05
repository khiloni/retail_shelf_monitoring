"""Planogram grid view widget."""
from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ....entities.planogram import Planogram


class PlanogramView(QWidget):
    """Visualizes the stored planogram structure."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.table = QTableWidget()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table)

    def set_planogram(self, planogram: Optional[Planogram]) -> None:
        if not planogram:
            self.table.clear()
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            return

        max_cols = max(len(row.items) for row in planogram.grid.rows) if planogram.grid.rows else 0
        self.table.setRowCount(len(planogram.grid.rows))
        self.table.setColumnCount(max_cols)
        self.table.setVerticalHeaderLabels([f"Row {r.row_idx}" for r in planogram.grid.rows])
        self.table.setHorizontalHeaderLabels([f"Slot {c}" for c in range(max_cols)])

        for r_idx, row in enumerate(planogram.grid.rows):
            for c_idx, item in enumerate(row.items):
                item_widget = QTableWidgetItem(item.sku_id)
                self.table.setItem(r_idx, c_idx, item_widget)
