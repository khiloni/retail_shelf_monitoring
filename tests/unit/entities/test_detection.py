"""Unit tests for Detection entity."""
from datetime import datetime, timezone
from shelf_monitor.entities.common import BoundingBox
from shelf_monitor.entities.detection import Detection


def test_detection_entity():
    now = datetime.now(timezone.utc)
    det = Detection(
        shelf_id="shelf_A",
        frame_timestamp=now,
        bbox=BoundingBox(x1=10, y1=10, x2=50, y2=50),
        class_id=0,
        sku_id="sku_cereal",
        confidence=0.88,
    )
    assert det.shelf_id == "shelf_A"
    assert det.confidence == 0.88
    assert det.sku_id == "sku_cereal"
