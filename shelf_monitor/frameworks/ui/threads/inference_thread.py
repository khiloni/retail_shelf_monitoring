"""Inference worker thread."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from queue import Empty, Queue
from typing import Any, List, Optional

import numpy as np
from PySide6.QtCore import QThread, Signal

from ....entities.detection import Detection
from ....frameworks.logging_config import get_logger
from ....usecases.stream_processing import StreamProcessingUseCase

logger = get_logger(__name__)


class InferenceThread(QThread):
    """Processes video frames through StreamProcessingUseCase in a background thread."""

    detections_ready = Signal(object, list, list, dict, float)
    # args: (frame_img, detections, cell_states, summary, fps)

    planogram_ready = Signal(object)
    # args: (Planogram,) — emitted when a planogram is first loaded for the active shelf

    def __init__(self, stream_processing_usecase: StreamProcessingUseCase) -> None:
        super().__init__()
        self.stream_processing = stream_processing_usecase
        self._queue: Queue[tuple[np.ndarray, int]] = Queue(maxsize=10)
        self._running = False
        self._last_planogram_shelf_id: Optional[str] = None

    def enqueue_frame(self, frame_img: np.ndarray, frame_idx: int) -> None:
        if self._running:
            if self._queue.full():
                try:
                    self._queue.get_nowait()  # Drop oldest frame to maintain low latency
                except Empty:
                    pass
            self._queue.put_nowait((frame_img, frame_idx))

    def run(self) -> None:
        self._running = True
        logger.info("Inference thread started")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        t_last = time_now = datetime.now(timezone.utc).timestamp()
        frame_count = 0

        while self._running:
            try:
                frame_img, frame_idx = self._queue.get(timeout=0.2)
            except Empty:
                continue

            ts = datetime.now(timezone.utc)
            frame_id = f"frame_{frame_idx}"

            try:
                result = loop.run_until_complete(
                    self.stream_processing.process_frame(frame_img, frame_id, ts)
                )

                frame_count += 1
                now = ts.timestamp()
                fps = round(frame_count / max(0.001, (now - t_last)), 1)
                if now - t_last > 2.0:
                    t_last = now
                    frame_count = 0

                # Notify UI when a planogram is loaded for the current shelf
                shelf_id = self.stream_processing.fixed_shelf_id
                planogram = self.stream_processing._planograms.get(shelf_id)
                if planogram is not None and shelf_id != self._last_planogram_shelf_id:
                    self._last_planogram_shelf_id = shelf_id
                    self.planogram_ready.emit(planogram)

                self.detections_ready.emit(
                    frame_img,
                    result.detections,
                    result.cell_states,
                    result.summary or {},
                    fps,
                )
            except Exception as e:
                logger.error(f"Error in inference thread: {e}", exc_info=True)

        loop.close()
        logger.info("Inference thread stopped")

    def stop(self) -> None:
        self._running = False
        self.wait(1000)

