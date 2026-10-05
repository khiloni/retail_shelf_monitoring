"""Unit tests for cell state computation use case."""
from datetime import datetime, timezone
from shelf_monitor.entities.common import BoundingBox, CellState
from shelf_monitor.entities.detection import Detection
from shelf_monitor.entities.planogram import Planogram, PlanogramGrid, PlanogramItem, PlanogramRow
from shelf_monitor.usecases.cell_state_computation import CellStateComputation
from shelf_monitor.usecases.grid.grid_detector import GridDetector


def test_cell_state_ok_and_empty():
    item0 = PlanogramItem(item_idx=0, bbox=BoundingBox(x1=10, y1=10, x2=40, y2=50), sku_id="sku_A")
    item1 = PlanogramItem(item_idx=1, bbox=BoundingBox(x1=50, y1=10, x2=80, y2=50), sku_id="sku_B")
    row0 = PlanogramRow(row_idx=0, avg_y=30.0, items=[item0, item1])
    grid = PlanogramGrid(rows=[row0])
    planogram = Planogram(shelf_id="S1", grid=grid)

    grid_detector = GridDetector(clustering_method="dbscan", eps=20.0, min_samples=1)
    comp = CellStateComputation(grid_detector=grid_detector, position_tolerance=1)

    # Supply detection matching only item0
    now = datetime.now(timezone.utc)
    det0 = Detection(
        shelf_id="S1",
        frame_timestamp=now,
        bbox=BoundingBox(x1=10, y1=10, x2=40, y2=50),
        class_id=0,
        sku_id="sku_A",
        confidence=0.9,
    )

    res = comp.compute_cell_states(planogram, [det0], frame_timestamp=now)
    states = res["cell_states"]
    summary = res["summary"]

    assert summary["total_cells"] == 2
    assert summary["ok_count"] == 1
    assert summary["empty_count"] == 1
    assert summary["fill_pct"] == 50.0
    assert summary["compliance_pct"] == 50.0
