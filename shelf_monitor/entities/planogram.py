"""Planogram entity — the reference shelf layout."""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from .common import BoundingBox


class PlanogramItem(BaseModel):
    """A single slot in the planogram grid."""
    item_idx: int = Field(..., ge=0, description="Item index within its row")
    bbox: BoundingBox = Field(..., description="Bounding box in reference coordinates")
    sku_id: str = Field(..., description="Expected SKU at this position")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    track_id: Optional[int] = Field(default=None)

    model_config = {"frozen": True}


class PlanogramRow(BaseModel):
    """One shelf row in the planogram (top-to-bottom ordering)."""
    row_idx: int = Field(..., ge=0, description="Row index (0 = topmost)")
    avg_y: float = Field(..., ge=0.0, description="Average Y-centre of this row")
    items: List[PlanogramItem] = Field(..., min_length=1, description="Items left-to-right")

    @field_validator("items")
    @classmethod
    def items_sorted_left_to_right(cls, v: List[PlanogramItem]) -> List[PlanogramItem]:
        return sorted(v, key=lambda item: item.bbox.x1)

    model_config = {"frozen": True}


class PlanogramGrid(BaseModel):
    """Full grid of rows."""
    rows: List[PlanogramRow] = Field(..., min_length=1, description="Rows top-to-bottom")

    @field_validator("rows")
    @classmethod
    def rows_sorted_top_to_bottom(cls, v: List[PlanogramRow]) -> List[PlanogramRow]:
        return sorted(v, key=lambda row: row.avg_y)

    @property
    def total_items(self) -> int:
        return sum(len(row.items) for row in self.rows)

    def get_cell(self, row_idx: int, item_idx: int) -> Optional[PlanogramItem]:
        """Return the PlanogramItem at (row_idx, item_idx) or None."""
        for row in self.rows:
            if row.row_idx == row_idx:
                for item in row.items:
                    if item.item_idx == item_idx:
                        return item
        return None

    model_config = {"frozen": True}


class ClusteringParams(BaseModel):
    """Parameters used to cluster detections into rows."""
    row_clustering_method: str = Field(default="dbscan")
    eps: float = Field(default=15.0, gt=0.0)
    min_samples: int = Field(default=2, ge=1)

    model_config = {"frozen": True}


class Planogram(BaseModel):
    """The full planogram for one shelf."""
    shelf_id: str = Field(..., description="Associated shelf identifier")
    reference_image_path: str = Field(default="", description="Path to reference image")
    grid: PlanogramGrid
    clustering_params: ClusteringParams = Field(default_factory=ClusteringParams)
    meta: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {
        "frozen": False,
        "json_encoders": {datetime: lambda v: v.isoformat()},
    }
