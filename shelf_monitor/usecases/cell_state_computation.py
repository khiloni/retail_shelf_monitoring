"""Cell state computation usecase.

Compares current detections against reference planogram using GridDetector.
Produces per-cell states: OK, EMPTY, MISPLACED, UNKNOWN.
Computes summary metrics: ok, empty, misplaced, unknown, total_cells, fill_pct, compliance_pct.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..entities.common import CellState
from ..entities.detection import Detection
from ..entities.planogram import Planogram
from ..frameworks.logging_config import get_logger
from .grid.grid_detector import GridDetector

logger = get_logger(__name__)


class CellStateComputation:
    """Computes per-cell compliance states and summary statistics."""

    def __init__(
        self,
        grid_detector: GridDetector,
        position_tolerance: int = 1,
        confidence_threshold: float = 0.35,
    ) -> None:
        self.grid_detector = grid_detector
        self.position_tolerance = position_tolerance
        self.confidence_threshold = confidence_threshold

    def compute_cell_states(
        self,
        planogram: Planogram,
        detections: List[Detection],
        frame_timestamp: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Compare current detections against reference planogram.

        Returns:
            {
                "cell_states": List[Dict],
                "summary": {
                    "ok_count": int,
                    "empty_count": int,
                    "misplaced_count": int,
                    "unknown_count": int,
                    "total_cells": int,
                    "fill_pct": float,
                    "compliance_pct": float,
                    "timestamp": datetime,
                }
            }
        """
        ts = frame_timestamp or datetime.now(timezone.utc)

        detection_dicts = [
            {
                "bbox": [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2],
                "sku_id": d.sku_id or f"class_{d.class_id}",
                "confidence": d.confidence,
                "class_id": d.class_id,
                "track_id": d.track_id,
            }
            for d in detections
            if d.confidence >= self.confidence_threshold
        ]

        if not detection_dicts:
            return self._all_cells_empty(planogram, ts)

        match_result = self.grid_detector.match_grids(
            reference_grid=planogram.grid,
            current_detections=detection_dicts,
            position_tolerance=self.position_tolerance,
        )

        cell_states: List[Dict[str, Any]] = []

        for m in match_result["matches"]:
            cell_states.append(
                {
                    "row_idx": m["row_idx"],
                    "item_idx": m["item_idx"],
                    "state": CellState.OK,
                    "expected_sku": m["sku_id"],
                    "detected_sku": m["sku_id"],
                    "confidence": 1.0,
                }
            )

        for mm in match_result["mismatches"]:
            det_sku = mm.get("detected_sku")
            # If detected SKU is unknown or missing, mark accordingly
            state = CellState.UNKNOWN if det_sku == "unknown_sku" else CellState.MISPLACED
            cell_states.append(
                {
                    "row_idx": mm["row_idx"],
                    "item_idx": mm["item_idx"],
                    "state": state,
                    "expected_sku": mm["expected_sku"],
                    "detected_sku": det_sku,
                    "confidence": 1.0,
                    "track_id": mm.get("track_id"),
                }
            )

        for miss in match_result["missing"]:
            cell_states.append(
                {
                    "row_idx": miss["row_idx"],
                    "item_idx": miss["item_idx"],
                    "state": CellState.EMPTY,
                    "expected_sku": miss["expected_sku"],
                    "detected_sku": None,
                    "confidence": 0.0,
                }
            )

        total_cells = planogram.grid.total_items
        ok_count = sum(1 for c in cell_states if c["state"] == CellState.OK)
        empty_count = sum(1 for c in cell_states if c["state"] == CellState.EMPTY)
        misplaced_count = sum(1 for c in cell_states if c["state"] == CellState.MISPLACED)
        unknown_count = sum(1 for c in cell_states if c["state"] == CellState.UNKNOWN)

        # Metrics
        present_count = total_cells - empty_count
        fill_pct = (present_count / total_cells * 100.0) if total_cells > 0 else 0.0
        compliance_pct = (ok_count / total_cells * 100.0) if total_cells > 0 else 0.0

        summary = {
            "ok_count": ok_count,
            "empty_count": empty_count,
            "misplaced_count": misplaced_count,
            "unknown_count": unknown_count,
            "total_cells": total_cells,
            "fill_pct": round(fill_pct, 2),
            "compliance_pct": round(compliance_pct, 2),
            "timestamp": ts.isoformat(),
        }

        logger.debug(
            f"Cell state computation ({planogram.shelf_id}): "
            f"OK={ok_count}, EMPTY={empty_count}, MISPLACED={misplaced_count}, UNKNOWN={unknown_count}"
        )

        return {"cell_states": cell_states, "summary": summary}

    def _all_cells_empty(self, planogram: Planogram, ts: datetime) -> Dict[str, Any]:
        cell_states = []
        for row in planogram.grid.rows:
            for item in row.items:
                cell_states.append(
                    {
                        "row_idx": row.row_idx,
                        "item_idx": item.item_idx,
                        "state": CellState.EMPTY,
                        "expected_sku": item.sku_id,
                        "detected_sku": None,
                        "confidence": 0.0,
                    }
                )

        total = planogram.grid.total_items
        summary = {
            "ok_count": 0,
            "empty_count": total,
            "misplaced_count": 0,
            "unknown_count": 0,
            "total_cells": total,
            "fill_pct": 0.0,
            "compliance_pct": 0.0,
            "timestamp": ts.isoformat(),
        }
        return {"cell_states": cell_states, "summary": summary}
