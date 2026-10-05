"""Alert analysis worker thread."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from queue import Empty, Queue
from typing import Any, Dict, List

from PySide6.QtCore import QThread, Signal

from ....frameworks.logging_config import get_logger
from ....usecases.alert_generation import AlertGenerationUseCase

logger = get_logger(__name__)


class AlertAnalysisThread(QThread):
    """Processes temporal consensus alerts and creates persistent records."""

    alerts_updated = Signal()

    def __init__(self, alert_generation_usecase: AlertGenerationUseCase) -> None:
        super().__init__()
        self.alert_gen = alert_generation_usecase
        self._queue: Queue[Dict[str, Any]] = Queue(maxsize=50)
        self._running = False

    def enqueue_analysis(self, alert_data: Dict[str, Any]) -> None:
        if self._running and not self._queue.full():
            self._queue.put_nowait(alert_data)

    def run(self) -> None:
        self._running = True
        logger.info("Alert analysis thread started")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        while self._running:
            try:
                data = self._queue.get(timeout=0.2)
            except Empty:
                continue

            try:
                loop.run_until_complete(self.alert_gen.generate_alert(data))
                self.alerts_updated.emit()
            except Exception as e:
                logger.error(f"Alert analysis error: {e}")

        loop.close()
        logger.info("Alert analysis thread stopped")

    def stop(self) -> None:
        self._running = False
        self.wait(1000)
