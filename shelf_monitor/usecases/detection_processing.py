"""Detection processing usecase.

Runs YOLO detector on a frame, crops detected bounding boxes,
passes crops to the SKU recognizer, and produces Detection entities.
"""
from __future__ import annotations

import uuid
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

from ..entities.common import BoundingBox
from ..entities.detection import Detection
from ..entities.frame import Frame
from ..frameworks.logging_config import get_logger

logger = get_logger(__name__)


class DetectionProcessingUseCase:
    """Orchestrates detection + SKU identification on a single frame."""

    def __init__(self, detector: Any, sku_recognizer: Optional[Any] = None) -> None:
        self.detector = detector
        self.sku_recognizer = sku_recognizer

    def process_aligned_frame(self, frame: Frame) -> List[Detection]:
        """Detect products and identify their SKUs in an aligned frame."""
        # 1. Run object detection
        raw_detections = self.detector.predict(frame.frame_img)
        if not raw_detections:
            return []

        h, w = frame.frame_img.shape[:2]

        # 2. If SKU recognizer is available and in FULL mode, identify SKUs
        if self.sku_recognizer is not None and getattr(self.sku_recognizer, "has_index", True):
            crops = []
            valid_indices = []
            for i, det in enumerate(raw_detections):
                bbox = det.bbox if hasattr(det, "bbox") else det["bbox"]
                if hasattr(bbox, "x1"):
                    x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
                else:
                    x1, y1, x2, y2 = [int(v) for v in bbox]

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 > x1 and y2 > y1:
                    crop = frame.frame_img[y1:y2, x1:x2]
                    crops.append(crop)
                    valid_indices.append(i)

            if crops:
                sku_ids = self.sku_recognizer.batch_identify_skus(crops)
                for idx, sku_id in zip(valid_indices, sku_ids):
                    if hasattr(raw_detections[idx], "sku_id"):
                        raw_detections[idx].sku_id = sku_id
                    elif isinstance(raw_detections[idx], dict):
                        raw_detections[idx]["sku_id"] = sku_id

        # 3. Ensure all are Detection entities
        result: List[Detection] = []
        for det in raw_detections:
            if isinstance(det, Detection):
                det.shelf_id = frame.shelf_id or "default_shelf"
                det.frame_timestamp = frame.timestamp
                result.append(det)
            else:
                bbox_raw = det["bbox"]
                bbox = (
                    bbox_raw
                    if isinstance(bbox_raw, BoundingBox)
                    else BoundingBox(
                        x1=bbox_raw[0], y1=bbox_raw[1],
                        x2=bbox_raw[2], y2=bbox_raw[3]
                    )
                )
                result.append(
                    Detection(
                        detection_id=str(uuid.uuid4()),
                        shelf_id=frame.shelf_id or "default_shelf",
                        frame_timestamp=frame.timestamp,
                        bbox=bbox,
                        class_id=det.get("class_id", 0),
                        sku_id=det.get("sku_id", "unknown_sku"),
                        confidence=float(det.get("confidence", 1.0)),
                        track_id=det.get("track_id"),
                    )
                )

        return result
