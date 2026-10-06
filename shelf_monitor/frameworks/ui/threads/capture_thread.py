"""Capture worker thread for video/camera feeds."""
from __future__ import annotations

import time
from typing import Optional, Union

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from ....frameworks.logging_config import get_logger

logger = get_logger(__name__)


class CaptureThread(QThread):
    """Reads frames from a video file, webcam index, or RTSP stream."""

    frame_captured = Signal(object, int)  # frame_img (np.ndarray), frame_number
    capture_finished = Signal()
    error_occurred = Signal(str)

    def __init__(self, source: Union[str, int] = 0, target_fps: float = 30.0) -> None:
        super().__init__()
        self.source = source
        self.target_fps = target_fps
        self._running = False

    def run(self) -> None:
        self._running = True
        cap = cv2.VideoCapture(self.source)

        if not cap.isOpened():
            self.error_occurred.emit(f"Could not open video source: {self.source}")
            self._running = False
            return

        frame_interval = 1.0 / max(1.0, self.target_fps)
        frame_idx = 0

        logger.info(f"Capture thread started for source: {self.source}")

        while self._running:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.info("Video capture reached end of stream")
                break

            self.frame_captured.emit(frame, frame_idx)
            frame_idx += 1

            elapsed = time.time() - t0
            sleep_time = max(0.001, frame_interval - elapsed)
            time.sleep(sleep_time)

        cap.release()
        self.capture_finished.emit()
        logger.info("Capture thread stopped")

    def stop(self) -> None:
        self._running = False
        self.wait(1000)
