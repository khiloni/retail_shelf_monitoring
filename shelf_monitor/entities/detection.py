"""Detection entity — one detected product in a frame."""
import uuid
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from .common import BoundingBox


class Detection(BaseModel):
    """A single product detection from the YOLO detector."""
    detection_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    shelf_id: str = Field(..., description="Shelf this detection belongs to")
    frame_timestamp: datetime = Field(..., description="Frame capture timestamp")
    bbox: BoundingBox = Field(..., description="Bounding box in image coordinates")
    class_id: int = Field(default=0, ge=0, description="YOLO class id (0 = product)")
    sku_id: Optional[str] = Field(default=None, description="Identified SKU or None")
    confidence: float = Field(..., ge=0.0, le=1.0)
    track_id: Optional[int] = Field(default=None)
    row_idx: Optional[int] = Field(default=None, ge=0)
    item_idx: Optional[int] = Field(default=None, ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        # Accept very low confidence from trackers / synthetic data
        return max(0.0, min(1.0, float(v)))

    model_config = {
        "frozen": False,
        "json_encoders": {datetime: lambda v: v.isoformat()},
    }
