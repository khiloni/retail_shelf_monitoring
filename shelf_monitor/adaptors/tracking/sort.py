"""SORT (Simple Online and Realtime Tracking) implementation.

Uses a Kalman filter for state estimation and Hungarian algorithm (scipy linear_sum_assignment)
for data association via IoU matching.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from ...entities.common import BoundingBox
from ...entities.detection import Detection
from ...frameworks.logging_config import get_logger
from ...usecases.interfaces.tracker import Tracker

logger = get_logger(__name__)


def iou_bbox(b1: list[float] | np.ndarray, b2: list[float] | np.ndarray) -> float:
    """IoU for [x1, y1, x2, y2]."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return float(inter / union) if union > 0 else 0.0


def convert_bbox_to_z(bbox: list[float]) -> np.ndarray:
    """[x1, y1, x2, y2] -> [cx, cy, s, r] where s=area, r=w/h."""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    s = w * h
    r = w / float(h + 1e-6)
    return np.array([cx, cy, s, r]).reshape((4, 1))


def convert_x_to_bbox(x: np.ndarray) -> np.ndarray:
    """State [cx, cy, s, r] -> [x1, y1, x2, y2]."""
    cx, cy, s, r = x[0], x[1], x[2], x[3]
    w = np.sqrt(max(0.0, s * r))
    h = s / (w + 1e-6)
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]).reshape((1, 4))


class KalmanBoxTracker:
    """Kalman filter for a single bounding box."""

    count = 0

    def __init__(self, detection: Detection, max_w: float = 4096, max_h: float = 4096) -> None:
        self.detection = detection
        self.max_w = max_w
        self.max_h = max_h

        bbox = [detection.bbox.x1, detection.bbox.y1, detection.bbox.x2, detection.bbox.y2]
        self._x = np.zeros((7, 1))
        self._x[0:4] = convert_bbox_to_z(bbox)
        self._P = np.eye(7) * 10.0
        self._F = np.eye(7)
        self._H = np.zeros((4, 7))
        self._H[0, 0] = self._H[1, 1] = self._H[2, 2] = self._H[3, 3] = 1.0
        self._R = np.eye(4) * 1.0
        self._Q = np.eye(7) * 0.01

        self.time_since_update = 0
        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        self.hits = 1
        self.age = 0

    def predict(self) -> np.ndarray:
        F = np.eye(7)
        F[0, 4] = F[1, 5] = F[2, 6] = 1.0
        self._x = F @ self._x
        self._P = F @ self._P @ F.T + self._Q
        self.age += 1
        self.time_since_update += 1
        return convert_x_to_bbox(self._x[:4].flatten())

    def update(self, detection: Detection) -> None:
        bbox = [detection.bbox.x1, detection.bbox.y1, detection.bbox.x2, detection.bbox.y2]
        z = convert_bbox_to_z(bbox)
        self.detection = detection
        self.detection.track_id = self.id

        S = self._H @ self._P @ self._H.T + self._R
        K = self._P @ self._H.T @ np.linalg.inv(S)
        y = z - (self._H @ self._x)
        self._x = self._x + K @ y
        I = np.eye(7)
        self._P = (I - K @ self._H) @ self._P
        self.time_since_update = 0
        self.hits += 1

    def get_state(self) -> np.ndarray:
        return convert_x_to_bbox(self._x.flatten())

    def get_detection(self) -> Detection:
        b = convert_x_to_bbox(self._x.flatten()).flatten()
        det = self.detection.model_copy(deep=True)
        det.bbox = BoundingBox(x1=float(b[0]), y1=float(b[1]), x2=float(b[2]), y2=float(b[3]))
        det.track_id = self.id
        return det


class SortTracker(Tracker):
    """SORT multi-object tracker."""

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        max_bbox_width: float = 4096,
        max_bbox_height: float = 4096,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.max_w = max_bbox_width
        self.max_h = max_bbox_height
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count: int = 0

    def update(self, detections: List[Detection]) -> List[Detection]:
        self.frame_count += 1
        for trk in self.trackers:
            trk.predict()

        if not self.trackers:
            for det in detections:
                trk = KalmanBoxTracker(det, self.max_w, self.max_h)
                self.trackers.append(trk)
                det.track_id = trk.id
            return detections

        dets_arr = np.array([[d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2] for d in detections]) if detections else np.empty((0, 4))
        trks_arr = np.array([trk.get_state().flatten() for trk in self.trackers]).reshape(-1, 4)

        if len(dets_arr) > 0 and len(trks_arr) > 0:
            iou_matrix = np.zeros((len(dets_arr), len(trks_arr)), dtype=np.float32)
            for d_idx, d_box in enumerate(dets_arr):
                for t_idx, t_box in enumerate(trks_arr):
                    iou_matrix[d_idx, t_idx] = iou_bbox(d_box, t_box)

            cost = 1.0 - iou_matrix
            row_ind, col_ind = linear_sum_assignment(cost)

            matched_dets = set()
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= self.iou_threshold:
                    self.trackers[c].update(detections[r])
                    detections[r].track_id = self.trackers[c].id
                    matched_dets.add(r)

            # Create new trackers for unmatched detections
            for i, det in enumerate(detections):
                if i not in matched_dets:
                    trk = KalmanBoxTracker(det, self.max_w, self.max_h)
                    self.trackers.append(trk)
                    det.track_id = trk.id
        else:
            for det in detections:
                trk = KalmanBoxTracker(det, self.max_w, self.max_h)
                self.trackers.append(trk)
                det.track_id = trk.id

        # Prune old trackers
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]
        return detections

    def predict(self) -> List[Detection]:
        predicted = []
        for trk in self.trackers:
            try:
                trk.predict()
                if trk.time_since_update <= self.max_age:
                    predicted.append(trk.get_detection())
            except Exception:
                continue
        return predicted

    def reset(self) -> None:
        self.trackers.clear()
        self.frame_count = 0
