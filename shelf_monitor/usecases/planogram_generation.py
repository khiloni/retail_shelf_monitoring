"""Planogram generation use case.

Creates a Planogram from a reference shelf image by running product detection,
SKU identification, and row/column clustering, then stores it in the repository.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import cv2

from ..entities.common import BoundingBox
from ..entities.detection import Detection
from ..entities.frame import Frame
from ..entities.planogram import Planogram
from ..frameworks.exceptions import ValidationError
from ..frameworks.logging_config import get_logger
from .grid.grid_detector import GridDetector
from .interfaces.repositories import PlanogramRepository

logger = get_logger(__name__)


class PlanogramGenerationUseCase:
    """Generates and persists a planogram from a reference image."""

    def __init__(
        self,
        planogram_repository: PlanogramRepository,
        detector: Any,
        sku_recognizer: Optional[Any],
        grid_detector: GridDetector,
    ) -> None:
        self.planogram_repository = planogram_repository
        self.detector = detector
        self.sku_recognizer = sku_recognizer
        self.grid_detector = grid_detector

    async def generate_planogram_from_reference(
        self, shelf_id: str, reference_image_path: str | Path
    ) -> Planogram:
        image_path = Path(reference_image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Reference image not found: {reference_image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValidationError(f"Failed to read image: {reference_image_path}")

        logger.info(f"Generating planogram for shelf '{shelf_id}' from {image_path.name}")

        # 1. Detect products
        raw_detections = self.detector.predict(image)
        if not raw_detections:
            raise ValidationError(f"No products detected in reference image: {image_path}")

        # 2. Extract crops and identify SKUs
        h, w = image.shape[:2]
        detection_dicts = []
        crops = []
        crop_indices = []

        for i, det in enumerate(raw_detections):
            if hasattr(det, "bbox"):
                bbox = [det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2]
                conf = det.confidence
                cls_id = det.class_id
            else:
                bbox = det["bbox"]
                conf = det.get("confidence", 1.0)
                cls_id = det.get("class_id", 0)

            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            d_entry = {
                "bbox": bbox,
                "confidence": conf,
                "class_id": cls_id,
                "sku_id": f"sku_{cls_id}",
            }
            detection_dicts.append(d_entry)

            if x2 > x1 and y2 > y1 and self.sku_recognizer is not None:
                crops.append(image[y1:y2, x1:x2])
                crop_indices.append(i)

        if crops and self.sku_recognizer is not None and getattr(self.sku_recognizer, "has_index", True):
            sku_ids = self.sku_recognizer.batch_identify_skus(crops)
            for idx, sku_id in zip(crop_indices, sku_ids):
                detection_dicts[idx]["sku_id"] = sku_id

        # 3. Detect grid
        grid, clustering_params = self.grid_detector.detect_grid(detection_dicts)

        planogram = Planogram(
            shelf_id=shelf_id,
            reference_image_path=str(image_path),
            grid=grid,
            clustering_params=clustering_params,
            meta={
                "total_items": grid.total_items,
                "total_rows": len(grid.rows),
                "image_width": w,
                "image_height": h,
            },
        )

        saved = await self.planogram_repository.create(planogram)
        logger.info(
            f"Saved planogram for shelf '{shelf_id}' with {len(grid.rows)} rows "
            f"and {grid.total_items} items"
        )
        return saved
