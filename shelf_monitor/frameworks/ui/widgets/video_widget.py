"""Video playback and detection overlay widget."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy, QWidget

from ....entities.common import CellState


class VideoWidget(QLabel):
    """Displays video frames with overlaid detection bounding boxes and cell states."""

    STATE_COLORS = {
        CellState.OK: QColor(40, 167, 69, 180),        # Green
        CellState.EMPTY: QColor(220, 53, 69, 180),     # Red
        CellState.MISPLACED: QColor(255, 193, 7, 180), # Amber
        CellState.UNKNOWN: QColor(108, 117, 125, 180), # Grey
    }

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(480, 270)
        self.setText("No video feed active")
        self.setStyleSheet("background-color: #121214; color: #5c6370; font-size: 14px;")

        self._current_frame: Optional[np.ndarray] = None
        self._detections: List[Any] = []
        self._cell_states: List[Dict[str, Any]] = []

    def update_frame(
        self,
        frame_img: np.ndarray,
        detections: Optional[List[Any]] = None,
        cell_states: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self._current_frame = frame_img
        self._detections = detections or []
        self._cell_states = cell_states or []

        # Convert OpenCV BGR -> RGB QImage
        h, w, ch = frame_img.shape
        bytes_per_line = ch * w
        rgb_img = np.ascontiguousarray(cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB))
        qimg = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()

        # Draw overlays
        pixmap = QPixmap.fromImage(qimg)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        font = QFont("Segoe UI", 9, QFont.Bold)
        painter.setFont(font)

        # 1. Draw detection boxes
        pen_det = QPen(QColor(97, 175, 239), 2)
        painter.setPen(pen_det)
        for det in self._detections:
            if hasattr(det, "bbox"):
                x1, y1 = int(det.bbox.x1), int(det.bbox.y1)
                x2, y2 = int(det.bbox.x2), int(det.bbox.y2)
                sku_str = getattr(det, "sku_id", "product") or "product"
            else:
                x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
                sku_str = det.get("sku_id", "product")

            painter.drawRect(x1, y1, x2 - x1, y2 - y1)
            painter.fillRect(x1, max(0, y1 - 18), len(sku_str) * 8 + 8, 18, QColor(0, 0, 0, 160))
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(x1 + 4, max(14, y1 - 4), sku_str)
            painter.setPen(pen_det)

        painter.end()

        # Scale to widget size preserving aspect ratio
        scaled = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)
        self.setText("")
