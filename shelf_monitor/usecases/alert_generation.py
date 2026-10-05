"""Alert generation and alert management use cases."""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from ..entities.alert import Alert
from ..frameworks.logging_config import get_logger
from .interfaces.repositories import AlertRepository, PlanogramRepository

logger = get_logger(__name__)


class AlertGenerationUseCase:
    """Creates, updates, and clears shelf alerts, with optional evidence image storage."""

    def __init__(
        self,
        alert_repository: AlertRepository,
        planogram_repository: PlanogramRepository,
        evidence_dir: str | Path = "outputs/evidence",
    ) -> None:
        self.alert_repository = alert_repository
        self.planogram_repository = planogram_repository
        self.evidence_dir = Path(evidence_dir)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)

    def save_evidence_image(
        self,
        frame_img: np.ndarray,
        shelf_id: str,
        row_idx: int,
        item_idx: int,
        alert_type: str,
        crop_bbox: Optional[list[float]] = None,
    ) -> str:
        """Save evidence crop or full frame to evidence_dir and return file path."""
        ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        img_to_save = frame_img
        if crop_bbox is not None and len(crop_bbox) == 4:
            x1, y1, x2, y2 = [int(v) for v in crop_bbox]
            h, w = frame_img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                img_to_save = frame_img[y1:y2, x1:x2]

        fname = f"{shelf_id}_r{row_idx}_i{item_idx}_{alert_type}_{ts_str}.jpg"
        target_path = self.evidence_dir / fname
        cv2.imwrite(str(target_path), img_to_save)
        return str(target_path)

    async def generate_alert(
        self, alert_data: Dict[str, Any], evidence_paths: Optional[List[str]] = None
    ) -> Alert:
        shelf_id = alert_data["shelf_id"]
        row_idx = alert_data["row_idx"]
        item_idx = alert_data["item_idx"]
        alert_type = alert_data["alert_type"]

        existing = await self.alert_repository.get_by_cell(
            shelf_id=shelf_id, row_idx=row_idx, item_idx=item_idx
        )

        now = datetime.now(timezone.utc)

        if existing and not existing.confirmed and not existing.dismissed:
            existing.last_seen = now
            existing.consecutive_frames = alert_data.get("consecutive_frames", existing.consecutive_frames + 1)
            if evidence_paths:
                existing.evidence_paths.extend(evidence_paths)
            return await self.alert_repository.update(existing)

        alert = Alert(
            alert_id=str(uuid.uuid4()),
            shelf_id=shelf_id,
            row_idx=row_idx,
            item_idx=item_idx,
            alert_type=alert_type,
            expected_sku=alert_data.get("expected_sku"),
            detected_sku=alert_data.get("detected_sku"),
            first_seen=now,
            last_seen=now,
            evidence_paths=evidence_paths or [],
            consecutive_frames=alert_data.get("consecutive_frames", 1),
        )

        return await self.alert_repository.create(alert)

    async def clear_cell_alerts(self, shelf_id: str, row_idx: int, item_idx: int) -> None:
        existing = await self.alert_repository.get_by_cell(
            shelf_id=shelf_id, row_idx=row_idx, item_idx=item_idx
        )
        if existing and not existing.confirmed and not existing.dismissed:
            await self.alert_repository.dismiss_alert(existing.alert_id)
            logger.info(f"Auto-dismissed alert {existing.alert_id} for cell ({row_idx}, {item_idx})")


class AlertManagementUseCase:
    """Handles staff confirmation and dismissal of alerts."""

    def __init__(self, alert_repository: AlertRepository) -> None:
        self.alert_repository = alert_repository

    async def get_active_alerts(self, shelf_id: Optional[str] = None) -> List[Alert]:
        return await self.alert_repository.get_active_alerts(shelf_id)

    async def get_alert_by_id(self, alert_id: str) -> Optional[Alert]:
        return await self.alert_repository.get_by_id(alert_id)

    async def confirm_alert(self, alert_id: str, confirmed_by: str) -> Alert:
        return await self.alert_repository.confirm_alert(alert_id, confirmed_by)

    async def dismiss_alert(self, alert_id: str) -> Alert:
        return await self.alert_repository.dismiss_alert(alert_id)
