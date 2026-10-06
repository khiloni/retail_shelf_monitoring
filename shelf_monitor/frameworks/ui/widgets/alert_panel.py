"""Alert panel widget with list, filter, confirm/dismiss actions, and evidence preview."""
from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ....entities.alert import Alert


class AlertPanel(QWidget):
    """Staff alert management panel."""

    alert_confirmed = Signal(str, str)  # alert_id, staff_id
    alert_dismissed = Signal(str)       # alert_id

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._alerts: List[Alert] = []
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Table
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["ID", "Shelf", "Cell", "Type", "Status"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.table)

        # Evidence Preview
        self.evidence_label = QLabel("No evidence selected")
        self.evidence_label.setAlignment(Qt.AlignCenter)
        self.evidence_label.setFixedHeight(120)
        self.evidence_label.setStyleSheet("border: 1px dashed #3e4451; color: #5c6370;")
        layout.addWidget(self.evidence_label)

        # Action Buttons
        btn_layout = QHBoxLayout()
        self.btn_confirm = QPushButton("Confirm")
        self.btn_confirm.setObjectName("btn_confirm")
        self.btn_confirm.clicked.connect(self._confirm_selected)

        self.btn_dismiss = QPushButton("Dismiss")
        self.btn_dismiss.setObjectName("btn_dismiss")
        self.btn_dismiss.clicked.connect(self._dismiss_selected)

        btn_layout.addWidget(self.btn_confirm)
        btn_layout.addWidget(self.btn_dismiss)
        layout.addLayout(btn_layout)

    def set_alerts(self, alerts: List[Alert]) -> None:
        self._alerts = alerts
        self.table.setRowCount(len(alerts))
        for row, a in enumerate(alerts):
            status = "Confirmed" if a.confirmed else ("Dismissed" if a.dismissed else "Active")
            self.table.setItem(row, 0, QTableWidgetItem(a.alert_id[:8]))
            self.table.setItem(row, 1, QTableWidgetItem(a.shelf_id))
            self.table.setItem(row, 2, QTableWidgetItem(f"R{a.row_idx} C{a.item_idx}"))
            self.table.setItem(row, 3, QTableWidgetItem(str(a.alert_type.value if hasattr(a.alert_type, "value") else a.alert_type)))
            self.table.setItem(row, 4, QTableWidgetItem(status))

    def _on_selection_changed(self) -> None:
        rows = self.table.selectedIndexes()
        if not rows:
            return
        row = rows[0].row()
        if 0 <= row < len(self._alerts):
            alert = self._alerts[row]
            t_str = alert.alert_type.value if hasattr(alert.alert_type, "value") else str(alert.alert_type)
            det_text = (
                f"[{t_str.upper()}] Shelf {alert.shelf_id} Cell (R{alert.row_idx}, C{alert.item_idx})\n"
                f"Expected: {alert.expected_sku or 'none'} | Detected: {alert.detected_sku or 'none'}"
            )
            if alert.evidence_paths:
                for ep in reversed(alert.evidence_paths):
                    from pathlib import Path
                    if Path(ep).exists():
                        pm = QPixmap(ep)
                        if not pm.isNull():
                            scaled = pm.scaled(self.evidence_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                            self.evidence_label.setPixmap(scaled)
                            self.evidence_label.setToolTip(det_text)
                            return
            self.evidence_label.setText(det_text)

    def _confirm_selected(self) -> None:
        rows = self.table.selectedIndexes()
        if not rows:
            return
        row = rows[0].row()
        if 0 <= row < len(self._alerts):
            self.alert_confirmed.emit(self._alerts[row].alert_id, "staff_user")

    def _dismiss_selected(self) -> None:
        rows = self.table.selectedIndexes()
        if not rows:
            return
        row = rows[0].row()
        if 0 <= row < len(self._alerts):
            self.alert_dismissed.emit(self._alerts[row].alert_id)
