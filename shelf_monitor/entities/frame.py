"""Frame entity — one captured video frame."""
import uuid
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class Frame(BaseModel):
    """One video frame flowing through the pipeline."""
    frame_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    frame_img: np.ndarray = Field(..., description="BGR image array")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_keyframe: bool = Field(default=False)
    shelf_id: Optional[str] = Field(default=None)
    alignment_confidence: float = Field(default=0.0)
    inlier_ratio: float = Field(default=0.0)

    model_config = ConfigDict(arbitrary_types_allowed=True)
