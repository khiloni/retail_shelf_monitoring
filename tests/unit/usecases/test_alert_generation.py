"""Unit tests for alert generation use case."""
import pytest
from shelf_monitor.adaptors.repositories.memory_alert_store import MemoryAlertStore
from shelf_monitor.entities.common import AlertType
from shelf_monitor.usecases.alert_generation import AlertGenerationUseCase, AlertManagementUseCase


@pytest.mark.asyncio
async def test_alert_generation_and_confirmation():
    alert_store = MemoryAlertStore()
    gen = AlertGenerationUseCase(alert_repository=alert_store, planogram_repository=None)
    mgmt = AlertManagementUseCase(alert_repository=alert_store)

    alert_data = {
        "shelf_id": "S1",
        "row_idx": 0,
        "item_idx": 1,
        "alert_type": AlertType.OOS,
        "expected_sku": "sku_cereal",
        "consecutive_frames": 3,
    }

    # Generate alert
    alert = await gen.generate_alert(alert_data)
    assert alert.shelf_id == "S1"
    assert alert.confirmed is False

    # Confirm alert
    confirmed = await mgmt.confirm_alert(alert.alert_id, confirmed_by="alice")
    assert confirmed.confirmed is True
    assert confirmed.confirmed_by == "alice"

    # Dismiss alert
    dismissed = await mgmt.dismiss_alert(alert.alert_id)
    assert dismissed.dismissed is True
