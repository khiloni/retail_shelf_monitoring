"""Unit tests for Alert entity."""
from datetime import datetime, timezone
import pytest
from shelf_monitor.entities.alert import Alert
from shelf_monitor.entities.common import AlertType


def test_alert_creation():
    now = datetime.now(timezone.utc)
    alert = Alert(
        shelf_id="shelf_1",
        row_idx=0,
        item_idx=1,
        alert_type=AlertType.OOS,
        expected_sku="sku_1",
        first_seen=now,
        last_seen=now,
    )
    assert alert.shelf_id == "shelf_1"
    assert alert.alert_type == AlertType.OOS
    assert alert.confirmed is False
    assert alert.dismissed is False


def test_alert_confirmation_validation():
    now = datetime.now(timezone.utc)
    with pytest.raises(ValueError):
        # confirmed_by set without confirmed=True must raise
        Alert(
            shelf_id="shelf_1",
            row_idx=0,
            item_idx=1,
            alert_type=AlertType.MISPLACED,
            first_seen=now,
            last_seen=now,
            confirmed=False,
            confirmed_by="staff_1",
        )
