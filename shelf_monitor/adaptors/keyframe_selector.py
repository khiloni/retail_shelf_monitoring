"""Keyframe selector adaptor."""
from __future__ import annotations

import cv2
import numpy as np
from typing import Optional

from ..entities.frame import Frame
from ..frameworks.logging_config import get_logger

logger = get_logger(__name__)


class KeyframeSelector:
    """Selects keyframes based on frame-to-frame visual differences."""

    def __init__(self, diff_threshold: float = 0.05) -> None:
        self.diff_threshold = diff_threshold
        self.last_keyframe: Optional[np.ndarray] = None

    def is_keyframe(self, frame: Frame) -> Frame:
        """Determines if the given frame differs significantly from the last keyframe."""
        if self.last_keyframe is None:
            self.last_keyframe = frame.frame_img.copy()
            frame.is_keyframe = True
            return frame

        diff = self._compute_diff(frame.frame_img, self.last_keyframe)
        if diff > self.diff_threshold:
            self.last_keyframe = frame.frame_img.copy()
            frame.is_keyframe = True
        else:
            frame.is_keyframe = False

        return frame

    def _compute_diff(self, f1: np.ndarray, f2: np.ndarray) -> float:
        g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY) if f1.ndim == 3 else f1
        g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY) if f2.ndim == 3 else f2
        diff = cv2.absdiff(g1, g2)
        return float(np.mean(diff) / 255.0)
