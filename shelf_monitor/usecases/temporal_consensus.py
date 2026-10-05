"""Temporal consensus manager for retail shelf state tracking.

Filters out noise, brief occlusions, and motion blur by requiring an issue
to persist for N consecutive keyframes before raising an alert, and M consecutive
clean frames before clearing it.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from ..entities.common import AlertType, CellState
from ..frameworks.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class CellTemporalState:
    shelf_id: str
    row_idx: int
    item_idx: int
    expected_sku: str

    current_state: CellState = CellState.UNKNOWN
    consecutive_empty_frames: int = 0
    consecutive_misplaced_frames: int = 0
    consecutive_ok_frames: int = 0

    last_detected_sku: Optional[str] = None
    last_update_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    state_history: List[CellState] = field(default_factory=list)
    max_history_length: int = 15

    def update(self, new_state: CellState, detected_sku: Optional[str] = None) -> None:
        self.last_update_time = datetime.now(timezone.utc)
        self.last_detected_sku = detected_sku

        self.state_history.append(new_state)
        if len(self.state_history) > self.max_history_length:
            self.state_history.pop(0)

        if new_state == CellState.EMPTY:
            self.consecutive_empty_frames += 1
            self.consecutive_misplaced_frames = 0
            self.consecutive_ok_frames = 0
        elif new_state == CellState.MISPLACED:
            self.consecutive_empty_frames = 0
            self.consecutive_misplaced_frames += 1
            self.consecutive_ok_frames = 0
        elif new_state == CellState.OK:
            self.consecutive_empty_frames = 0
            self.consecutive_misplaced_frames = 0
            self.consecutive_ok_frames += 1
        else:
            # UNKNOWN state - preserve counts or reset? Keep streak but do not increment
            pass

        self.current_state = new_state

    def is_oscillating(self, window: int = 5) -> bool:
        """Check if cell is rapidly flickering between states (e.g. motion/occlusion)."""
        if len(self.state_history) < window:
            return False
        recent = self.state_history[-window:]
        return len(set(recent)) > 2

    def should_trigger_oos_alert(self, n_confirm: int) -> bool:
        return self.consecutive_empty_frames >= n_confirm and not self.is_oscillating()

    def should_trigger_misplacement_alert(self, n_confirm: int) -> bool:
        return self.consecutive_misplaced_frames >= n_confirm and not self.is_oscillating()

    def should_clear_alert(self, n_clear: int) -> bool:
        return self.consecutive_ok_frames >= n_clear


class TemporalConsensusManager:
    """Manages temporal consensus across consecutive frames for each shelf cell."""

    def __init__(
        self,
        n_confirm: int = 3,
        n_clear: int = 2,
        state_timeout: timedelta = timedelta(minutes=5),
    ) -> None:
        self.n_confirm = n_confirm
        self.n_clear = n_clear
        self.state_timeout = state_timeout
        self.cell_states: Dict[str, Dict[Tuple[int, int], CellTemporalState]] = defaultdict(dict)

    def update_cell_states(
        self, shelf_id: str, cell_state_updates: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        new_alerts: List[Dict[str, Any]] = []
        cleared_alerts: List[Dict[str, Any]] = []

        for update in cell_state_updates:
            row_idx = update["row_idx"]
            item_idx = update["item_idx"]
            state = update["state"]
            expected_sku = update["expected_sku"]
            detected_sku = update.get("detected_sku")

            cell_key = (row_idx, item_idx)
            if cell_key not in self.cell_states[shelf_id]:
                self.cell_states[shelf_id][cell_key] = CellTemporalState(
                    shelf_id=shelf_id,
                    row_idx=row_idx,
                    item_idx=item_idx,
                    expected_sku=expected_sku,
                )

            cell = self.cell_states[shelf_id][cell_key]
            prev_state = cell.current_state
            cell.update(state, detected_sku)

            # Check for Out of Stock (EMPTY) trigger
            if cell.should_trigger_oos_alert(self.n_confirm):
                # Trigger when hitting the threshold or changing to EMPTY
                if prev_state != CellState.EMPTY or cell.consecutive_empty_frames == self.n_confirm:
                    new_alerts.append(
                        {
                            "shelf_id": shelf_id,
                            "row_idx": row_idx,
                            "item_idx": item_idx,
                            "expected_sku": expected_sku,
                            "detected_sku": None,
                            "consecutive_frames": cell.consecutive_empty_frames,
                            "alert_type": AlertType.OOS,
                        }
                    )

            # Check for Misplacement trigger
            if cell.should_trigger_misplacement_alert(self.n_confirm):
                if prev_state != CellState.MISPLACED or cell.consecutive_misplaced_frames == self.n_confirm:
                    new_alerts.append(
                        {
                            "shelf_id": shelf_id,
                            "row_idx": row_idx,
                            "item_idx": item_idx,
                            "expected_sku": expected_sku,
                            "detected_sku": detected_sku,
                            "consecutive_frames": cell.consecutive_misplaced_frames,
                            "alert_type": AlertType.MISPLACED,
                        }
                    )

            # Check for Clear trigger
            if cell.should_clear_alert(self.n_clear):
                hist_len = len(cell.state_history)
                w_start = -(self.n_confirm + self.n_clear)
                w_end = -self.n_clear

                had_alert = (
                    hist_len >= self.n_confirm + self.n_clear
                    and (
                        cell.state_history[w_start:w_end].count(CellState.EMPTY) >= self.n_confirm
                        or cell.state_history[w_start:w_end].count(CellState.MISPLACED) >= self.n_confirm
                    )
                )

                if had_alert:
                    cleared_alerts.append(
                        {
                            "shelf_id": shelf_id,
                            "row_idx": row_idx,
                            "item_idx": item_idx,
                        }
                    )

        self._cleanup_stale_states()

        if new_alerts or cleared_alerts:
            logger.info(
                f"Consensus ({shelf_id}): {len(new_alerts)} new alerts, {len(cleared_alerts)} cleared"
            )

        return {"new_alerts": new_alerts, "cleared_alerts": cleared_alerts}

    def _cleanup_stale_states(self) -> None:
        now = datetime.now(timezone.utc)
        for shelf_id in list(self.cell_states.keys()):
            for k in list(self.cell_states[shelf_id].keys()):
                if now - self.cell_states[shelf_id][k].last_update_time > self.state_timeout:
                    del self.cell_states[shelf_id][k]

    def get_cell_state(
        self, shelf_id: str, row_idx: int, item_idx: int
    ) -> Optional[CellTemporalState]:
        return self.cell_states.get(shelf_id, {}).get((row_idx, item_idx))
