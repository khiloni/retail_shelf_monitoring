"""Alert polling and notification worker thread."""
from __future__ import annotations

import asyncio
import time
from typing import List, Optional

from PySide6.QtCore import QThread, Signal

from ....entities.alert import Alert
from ....frameworks.logging_config import get_logger
from ....usecases.alert_generation import AlertManagementUseCase

logger = get_logger(__name__)


class AlertThread(QThread):
    """Periodically fetches active alerts and notifies UI."""

    active_alerts_fetched = Signal(list)  # List[Alert]

    def __init__(self, alert_management_usecase: AlertManagementUseCase, poll_interval: float = 1.0) -> None:
        super().__init__()
        self.alert_mgmt = alert_management_usecase
        self.poll_interval = poll_interval
        self._running = False
        self.current_shelf_id: Optional[str] = None

    def run(self) -> None:
        self._running = True
        logger.info("Alert thread started")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        while self._running:
            try:
                alerts = loop.run_until_complete(
                    self.alert_mgmt.get_active_alerts(self.current_shelf_id)
                )
                self.active_alerts_fetched.emit(alerts)
            except Exception as e:
                logger.error(f"Error fetching active alerts: {e}")

            elapsed = 0.0
            while self._running and elapsed < self.poll_interval:
                time.sleep(0.05)
                elapsed += 0.05

        loop.close()
        logger.info("Alert thread stopped")

    def stop(self) -> None:
        self._running = False
        self.wait(1000)
