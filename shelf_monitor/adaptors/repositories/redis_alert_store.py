"""Redis alert store (optional, fallback to MemoryAlertStore if redis not available)."""
from __future__ import annotations

import json
from typing import Any, List, Optional

from ...entities.alert import Alert
from ...frameworks.logging_config import get_logger
from ...usecases.interfaces.repositories import AlertRepository
from .memory_alert_store import MemoryAlertStore

logger = get_logger(__name__)


class RedisAlertStore(AlertRepository):
    """Stores alerts in Redis hash/strings. Falls back to in-memory store if Redis client fails."""

    def __init__(self, redis_client: Optional[Any] = None) -> None:
        self.client = redis_client
        self._fallback = MemoryAlertStore()
        self.prefix = "shelf_alert:"

    async def create(self, alert: Alert) -> Alert:
        if self.client is None:
            return await self._fallback.create(alert)
        try:
            key = f"{self.prefix}{alert.alert_id}"
            self.client.set(key, alert.model_dump_json())
            return alert
        except Exception as e:
            logger.warning(f"Redis write failed: {e}. Using fallback.")
            return await self._fallback.create(alert)

    async def get_by_id(self, alert_id: str) -> Optional[Alert]:
        if self.client is None:
            return await self._fallback.get_by_id(alert_id)
        try:
            data = self.client.get(f"{self.prefix}{alert_id}")
            return Alert.model_validate_json(data) if data else None
        except Exception:
            return await self._fallback.get_by_id(alert_id)

    async def get_by_cell(
        self, shelf_id: str, row_idx: int, item_idx: int
    ) -> Optional[Alert]:
        # Simple scan
        active = await self.get_active_alerts(shelf_id)
        for a in active:
            if a.row_idx == row_idx and a.item_idx == item_idx:
                return a
        return None

    async def update(self, alert: Alert) -> Alert:
        return await self.create(alert)

    async def get_active_alerts(self, shelf_id: Optional[str] = None) -> List[Alert]:
        if self.client is None:
            return await self._fallback.get_active_alerts(shelf_id)
        try:
            keys = self.client.keys(f"{self.prefix}*")
            alerts = []
            for k in keys:
                data = self.client.get(k)
                if data:
                    a = Alert.model_validate_json(data)
                    if not a.dismissed:
                        if shelf_id is None or a.shelf_id == shelf_id:
                            alerts.append(a)
            return alerts
        except Exception:
            return await self._fallback.get_active_alerts(shelf_id)

    async def confirm_alert(self, alert_id: str, confirmed_by: str) -> Alert:
        alert = await self.get_by_id(alert_id)
        if not alert:
            raise ValueError(f"Alert not found: {alert_id}")
        alert.confirmed = True
        alert.confirmed_by = confirmed_by
        return await self.update(alert)

    async def dismiss_alert(self, alert_id: str) -> Alert:
        alert = await self.get_by_id(alert_id)
        if not alert:
            raise ValueError(f"Alert not found: {alert_id}")
        alert.dismissed = True
        return await self.update(alert)
