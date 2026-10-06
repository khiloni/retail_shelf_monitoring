"""In-memory alert repository (default, no external Redis/DB required)."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from ...entities.alert import Alert
from ...frameworks.logging_config import get_logger
from ...usecases.interfaces.repositories import AlertRepository

logger = get_logger(__name__)


class MemoryAlertStore(AlertRepository):
    """Stores alerts in a thread-safe in-memory dictionary with disk persistence."""

    def __init__(self, persistence_path: Optional[str] = "outputs/runs/alerts.json") -> None:
        self._alerts: Dict[str, Alert] = {}
        self._lock = asyncio.Lock()
        self._persistence_path = Path(persistence_path) if persistence_path else None
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        if self._persistence_path and self._persistence_path.exists():
            try:
                import json
                with self._persistence_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                for item in data:
                    a = Alert.model_validate(item)
                    self._alerts[a.alert_id] = a
                logger.info(f"Loaded {len(self._alerts)} alerts from {self._persistence_path}")
            except Exception as e:
                logger.debug(f"Failed to load alerts from disk: {e}")

    def _save_to_disk(self) -> None:
        if self._persistence_path:
            try:
                import json
                self._persistence_path.parent.mkdir(parents=True, exist_ok=True)
                with self._persistence_path.open("w", encoding="utf-8") as f:
                    json.dump([a.model_dump(mode="json") for a in self._alerts.values()], f, indent=2)
            except Exception as e:
                logger.debug(f"Failed to persist alerts to disk: {e}")

    async def create(self, alert: Alert) -> Alert:
        async with self._lock:
            self._alerts[alert.alert_id] = alert
            self._save_to_disk()
        return alert

    async def get_by_id(self, alert_id: str) -> Optional[Alert]:
        async with self._lock:
            return self._alerts.get(alert_id)

    async def get_by_cell(
        self, shelf_id: str, row_idx: int, item_idx: int
    ) -> Optional[Alert]:
        async with self._lock:
            for a in self._alerts.values():
                if (
                    a.shelf_id == shelf_id
                    and a.row_idx == row_idx
                    and a.item_idx == item_idx
                    and not a.dismissed
                ):
                    return a
        return None

    async def update(self, alert: Alert) -> Alert:
        async with self._lock:
            alert.updated_at = datetime.now(timezone.utc)
            self._alerts[alert.alert_id] = alert
            self._save_to_disk()
        return alert

    async def get_active_alerts(self, shelf_id: Optional[str] = None) -> List[Alert]:
        async with self._lock:
            alerts = [a for a in self._alerts.values() if not a.dismissed]
            if shelf_id is not None:
                alerts = [a for a in alerts if a.shelf_id == shelf_id]
            return sorted(alerts, key=lambda a: a.last_seen, reverse=True)

    async def confirm_alert(self, alert_id: str, confirmed_by: str) -> Alert:
        async with self._lock:
            alert = self._alerts.get(alert_id)
            if not alert:
                raise ValueError(f"Alert not found: {alert_id}")
            alert.confirmed = True
            alert.confirmed_by = confirmed_by
            alert.confirmed_at = datetime.now(timezone.utc)
            alert.updated_at = datetime.now(timezone.utc)
            self._save_to_disk()
            return alert

    async def dismiss_alert(self, alert_id: str) -> Alert:
        async with self._lock:
            alert = self._alerts.get(alert_id)
            if not alert:
                raise ValueError(f"Alert not found: {alert_id}")
            alert.dismissed = True
            alert.updated_at = datetime.now(timezone.utc)
            self._save_to_disk()
            return alert

    def clear(self) -> None:
        self._alerts.clear()
        self._save_to_disk()
