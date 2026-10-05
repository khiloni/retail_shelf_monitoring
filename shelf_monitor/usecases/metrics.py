"""Shelf metrics computation and reporting."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List

from ..entities.common import CellState


@dataclass
class ShelfMetrics:
    total_cells: int
    product_count: int
    ok_count: int
    empty_count: int      # OOS count
    misplaced_count: int
    unknown_count: int
    fill_pct: float       # (product_count / total_cells) * 100
    compliance_pct: float # (ok_count / total_cells) * 100

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_metrics_from_cell_states(
    cell_states: List[Dict[str, Any]], total_cells: int
) -> ShelfMetrics:
    """Compute standard shelf metrics from a list of cell state dictionaries."""
    ok_c = sum(1 for c in cell_states if c["state"] == CellState.OK)
    empty_c = sum(1 for c in cell_states if c["state"] == CellState.EMPTY)
    misplaced_c = sum(1 for c in cell_states if c["state"] == CellState.MISPLACED)
    unknown_c = sum(1 for c in cell_states if c["state"] == CellState.UNKNOWN)

    product_count = total_cells - empty_c
    fill_pct = (product_count / total_cells * 100.0) if total_cells > 0 else 0.0
    comp_pct = (ok_c / total_cells * 100.0) if total_cells > 0 else 0.0

    return ShelfMetrics(
        total_cells=total_cells,
        product_count=product_count,
        ok_count=ok_c,
        empty_count=empty_c,
        misplaced_count=misplaced_c,
        unknown_count=unknown_c,
        fill_pct=round(fill_pct, 2),
        compliance_pct=round(comp_pct, 2),
    )
