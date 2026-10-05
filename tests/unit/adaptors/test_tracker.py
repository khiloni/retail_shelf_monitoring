"""Unit tests for SORT tracker."""
from datetime import datetime, timezone
from shelf_monitor.adaptors.tracking.sort import SortTracker
from shelf_monitor.entities.common import BoundingBox
from shelf_monitor.entities.detection import Detection


def test_sort_tracker_tracking():
    tracker = SortTracker(max_age=5, min_hits=1, iou_threshold=0.3)
    now = datetime.now(timezone.utc)

    # Frame 1: Detection at (10, 10, 50, 50)
    det1 = Detection(
        shelf_id="S1",
        frame_timestamp=now,
        bbox=BoundingBox(x1=10, y1=10, x2=50, y2=50),
        confidence=0.9,
    )
    tracked_1 = tracker.update([det1])
    assert len(tracked_1) == 1
    assert tracked_1[0].track_id is not None
    tid = tracked_1[0].track_id

    # Frame 2: Slightly moved box
    det2 = Detection(
        shelf_id="S1",
        frame_timestamp=now,
        bbox=BoundingBox(x1=12, y1=12, x2=52, y2=52),
        confidence=0.9,
    )
    tracked_2 = tracker.update([det2])
    assert len(tracked_2) == 1
    assert tracked_2[0].track_id == tid

    # Frame 3: Non-keyframe prediction
    preds = tracker.predict()
    assert len(preds) == 1
    assert preds[0].track_id == tid
