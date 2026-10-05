"""GridDetector: clusters detections into a PlanogramGrid and matches against reference."""
from __future__ import annotations

import difflib
from typing import Dict, List, Optional, Tuple

from ...entities.common import BoundingBox
from ...entities.planogram import (
    ClusteringParams,
    PlanogramGrid,
    PlanogramItem,
    PlanogramRow,
)
from ...frameworks.logging_config import get_logger
from .clustering import ClusterItem, ItemSorter, RowClusterer

logger = get_logger(__name__)


class GridDetector:
    """Build a PlanogramGrid from detections and compare it to a reference."""

    def __init__(
        self,
        clustering_method: str = "dbscan",
        eps: float = 15.0,
        min_samples: int = 2,
    ) -> None:
        self.clustering_params = ClusteringParams(
            row_clustering_method=clustering_method,
            eps=eps,
            min_samples=min_samples,
        )
        self._clusterer = RowClusterer(method=clustering_method, eps=eps, min_samples=min_samples)

    # ------------------------------------------------------------------
    # Grid detection
    # ------------------------------------------------------------------

    def detect_grid(
        self, detections: List[Dict]
    ) -> Tuple[PlanogramGrid, ClusteringParams]:
        """Build a PlanogramGrid from raw detection dicts."""
        if not detections:
            raise ValueError("Cannot build grid from empty detections")

        items = self._dicts_to_cluster_items(detections)
        row_clusters = self._clusterer.cluster_by_y_coordinate(items)

        if not row_clusters:
            raise ValueError("No valid row clusters found")

        rows: List[PlanogramRow] = []
        for row_idx, cluster in enumerate(row_clusters):
            avg_y = sum(it.center[1] for it in cluster) / len(cluster)
            indexed = ItemSorter.assign_indices(cluster)
            p_items = [
                PlanogramItem(
                    item_idx=idx,
                    bbox=BoundingBox(
                        x1=it.bbox[0], y1=it.bbox[1],
                        x2=it.bbox[2], y2=it.bbox[3],
                    ),
                    sku_id=it.sku_id,
                    confidence=it.confidence,
                    track_id=it.track_id,
                )
                for idx, it in indexed
            ]
            rows.append(PlanogramRow(row_idx=row_idx, avg_y=avg_y, items=p_items))

        grid = PlanogramGrid(rows=rows)
        logger.debug(f"Grid detected: {len(grid.rows)} rows, {grid.total_items} items")
        return grid, self.clustering_params

    # ------------------------------------------------------------------
    # Grid matching
    # ------------------------------------------------------------------

    def match_grids(
        self,
        reference_grid: PlanogramGrid,
        current_detections: List[Dict],
        position_tolerance: int = 1,
    ) -> Dict:
        """Compare current detections against the reference planogram.

        Returns a dict with keys: matches, mismatches, missing,
        reference_total, current_total.
        """
        if not current_detections:
            return self._all_missing(reference_grid)

        try:
            current_grid, _ = self.detect_grid(current_detections)
        except ValueError:
            return self._all_missing(reference_grid)

        matches, mismatches, missing = [], [], []

        for ref_row in reference_grid.rows:
            ri = ref_row.row_idx
            cur_row: Optional[PlanogramRow] = (
                current_grid.rows[ri] if ri < len(current_grid.rows) else None
            )
            if cur_row is None:
                for ref_item in ref_row.items:
                    missing.append(
                        {"row_idx": ri, "item_idx": ref_item.item_idx,
                         "expected_sku": ref_item.sku_id}
                    )
                continue

            r_m, r_mm, r_miss = self._match_rows(ref_row, cur_row, ri)
            matches.extend(r_m)
            mismatches.extend(r_mm)
            missing.extend(r_miss)

        return {
            "matches": matches,
            "mismatches": mismatches,
            "missing": missing,
            "reference_total": reference_grid.total_items,
            "current_total": current_grid.total_items,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dicts_to_cluster_items(self, detections: List[Dict]) -> List[ClusterItem]:
        items = []
        for d in detections:
            bbox = d["bbox"]
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            items.append(
                ClusterItem(
                    bbox=tuple(bbox),
                    center=(cx, cy),
                    sku_id=d.get("sku_id", "unknown_sku"),
                    confidence=d.get("confidence", 1.0),
                    track_id=d.get("track_id"),
                )
            )
        return items

    def _match_rows(
        self,
        ref_row: PlanogramRow,
        cur_row: PlanogramRow,
        row_idx: int,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        ref_seq = [it.sku_id for it in ref_row.items]
        cur_seq = [it.sku_id for it in cur_row.items]
        sm = difflib.SequenceMatcher(None, ref_seq, cur_seq)

        matches, mismatches, missing = [], [], []

        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                for k in range(i2 - i1):
                    matches.append(
                        {"row_idx": row_idx, "item_idx": i1 + k,
                         "sku_id": ref_seq[i1 + k]}
                    )
            elif tag == "replace":
                for k in range(min(i2 - i1, j2 - j1)):
                    mismatches.append(
                        {
                            "row_idx": row_idx,
                            "item_idx": i1 + k,
                            "expected_sku": ref_seq[i1 + k],
                            "detected_sku": cur_seq[j1 + k],
                            "track_id": cur_row.items[j1 + k].track_id
                            if j1 + k < len(cur_row.items) else None,
                        }
                    )
                for k in range(j2 - j1, i2 - i1):
                    missing.append(
                        {"row_idx": row_idx, "item_idx": i1 + k,
                         "expected_sku": ref_seq[i1 + k]}
                    )
            elif tag == "delete":
                for k in range(i2 - i1):
                    missing.append(
                        {"row_idx": row_idx, "item_idx": i1 + k,
                         "expected_sku": ref_seq[i1 + k]}
                    )

        return matches, mismatches, missing

    @staticmethod
    def _all_missing(reference_grid: PlanogramGrid) -> Dict:
        missing = []
        for row in reference_grid.rows:
            for item in row.items:
                missing.append(
                    {"row_idx": row.row_idx, "item_idx": item.item_idx,
                     "expected_sku": item.sku_id}
                )
        return {
            "matches": [], "mismatches": [], "missing": missing,
            "reference_total": reference_grid.total_items, "current_total": 0,
        }
