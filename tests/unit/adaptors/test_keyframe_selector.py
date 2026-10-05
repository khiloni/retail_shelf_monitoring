"""Unit tests for KeyframeSelector."""
from datetime import datetime, timezone
import numpy as np
from shelf_monitor.adaptors.keyframe_selector import KeyframeSelector
from shelf_monitor.entities.frame import Frame


def test_keyframe_selector_trigger():
    selector = KeyframeSelector(diff_threshold=0.1)
    now = datetime.now(timezone.utc)

    # Frame 1: first frame is always keyframe
    f1_img = np.zeros((100, 100, 3), dtype=np.uint8)
    f1 = Frame(frame_img=f1_img, timestamp=now)
    res1 = selector.is_keyframe(f1)
    assert res1.is_keyframe is True

    # Frame 2: identical image -> not a keyframe
    f2_img = np.zeros((100, 100, 3), dtype=np.uint8)
    f2 = Frame(frame_img=f2_img, timestamp=now)
    res2 = selector.is_keyframe(f2)
    assert res2.is_keyframe is False

    # Frame 3: very different image (bright white) -> keyframe!
    f3_img = np.full((100, 100, 3), 255, dtype=np.uint8)
    f3 = Frame(frame_img=f3_img, timestamp=now)
    res3 = selector.is_keyframe(f3)
    assert res3.is_keyframe is True
