"""Common enums and value objects for the shelf_monitor package."""
from enum import Enum
from typing import Tuple

from pydantic import BaseModel, Field, field_validator


class AlertType(str, Enum):
    """Types of shelf alerts."""
    OOS = "out_of_stock"          # Out-of-stock / empty cell
    MISPLACED = "misplaced"       # Wrong product in position
    UNKNOWN = "unknown"           # Cannot determine state


class CellState(str, Enum):
    """Single canonical set of per-cell states."""
    OK = "ok"
    EMPTY = "empty"               # alias for OOS
    MISPLACED = "misplaced"
    UNKNOWN = "unknown"


class BoundingBox(BaseModel):
    """Axis-aligned bounding box in pixel coordinates (top-left origin)."""
    x1: float = Field(..., description="Top-left X coordinate")
    y1: float = Field(..., description="Top-left Y coordinate")
    x2: float = Field(..., description="Bottom-right X coordinate")
    y2: float = Field(..., description="Bottom-right Y coordinate")

    @field_validator("x1", "y1", "x2", "y2", mode="before")
    @classmethod
    def clamp_non_negative(cls, v: float) -> float:
        return max(0.0, float(v))

    @property
    def center(self) -> Tuple[float, float]:
        """(cx, cy) centre of the box."""
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    model_config = {"frozen": True}
