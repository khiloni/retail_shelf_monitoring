"""Alert entity — a shelf compliance issue that needs staff attention."""
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator

from .common import AlertType


class Alert(BaseModel):
    """A shelf compliance alert (out-of-stock or misplaced product)."""
    alert_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    shelf_id: str = Field(..., description="Shelf this alert is for")
    row_idx: int = Field(..., ge=0)
    item_idx: int = Field(..., ge=0)
    alert_type: AlertType = Field(...)
    expected_sku: Optional[str] = Field(default=None)
    detected_sku: Optional[str] = Field(default=None)
    first_seen: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    confirmed: bool = Field(default=False)
    confirmed_by: Optional[str] = Field(default=None)
    confirmed_at: Optional[datetime] = Field(default=None)
    dismissed: bool = Field(default=False)
    evidence_paths: List[str] = Field(default_factory=list)
    consecutive_frames: int = Field(default=1, ge=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def confirm_fields_consistent(self) -> "Alert":
        if self.confirmed_at is not None and not self.confirmed:
            raise ValueError("confirmed_at can only be set when confirmed=True")
        if self.confirmed_by is not None and not self.confirmed:
            raise ValueError("confirmed_by can only be set when confirmed=True")
        return self

    model_config = {
        "frozen": False,
        "json_encoders": {datetime: lambda v: v.isoformat()},
    }
