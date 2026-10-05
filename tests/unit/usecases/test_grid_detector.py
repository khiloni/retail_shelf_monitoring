"""Unit tests for GridDetector clustering and matching."""
from shelf_monitor.entities.common import BoundingBox
from shelf_monitor.entities.planogram import PlanogramGrid, PlanogramItem, PlanogramRow
from shelf_monitor.usecases.grid.grid_detector import GridDetector


def test_grid_detector_detect_rows():
    detector = GridDetector(clustering_method="dbscan", eps=20.0, min_samples=1)

    detections = [
        {"bbox": [10, 20, 50, 60], "sku_id": "sku_1", "confidence": 0.9},
        {"bbox": [60, 22, 100, 62], "sku_id": "sku_2", "confidence": 0.95},
        {"bbox": [10, 150, 50, 190], "sku_id": "sku_3", "confidence": 0.9},
        {"bbox": [60, 152, 100, 192], "sku_id": "sku_4", "confidence": 0.9},
    ]

    grid, params = detector.detect_grid(detections)
    assert len(grid.rows) == 2
    assert grid.total_items == 4
    # First row items
    assert len(grid.rows[0].items) == 2
    assert grid.rows[0].items[0].sku_id == "sku_1"
    assert grid.rows[0].items[1].sku_id == "sku_2"


def test_grid_detector_match_grids():
    detector = GridDetector(clustering_method="dbscan", eps=20.0, min_samples=1)

    item0 = PlanogramItem(item_idx=0, bbox=BoundingBox(x1=10, y1=20, x2=50, y2=60), sku_id="sku_1")
    item1 = PlanogramItem(item_idx=1, bbox=BoundingBox(x1=60, y1=20, x2=100, y2=60), sku_id="sku_2")
    row0 = PlanogramRow(row_idx=0, avg_y=40.0, items=[item0, item1])
    ref_grid = PlanogramGrid(rows=[row0])

    # Case 1: Match
    res_match = detector.match_grids(
        reference_grid=ref_grid,
        current_detections=[
            {"bbox": [10, 20, 50, 60], "sku_id": "sku_1", "confidence": 0.9},
            {"bbox": [60, 20, 100, 60], "sku_id": "sku_2", "confidence": 0.9},
        ],
    )
    assert len(res_match["matches"]) == 2
    assert len(res_match["mismatches"]) == 0
    assert len(res_match["missing"]) == 0

    # Case 2: One missing
    res_missing = detector.match_grids(
        reference_grid=ref_grid,
        current_detections=[
            {"bbox": [10, 20, 50, 60], "sku_id": "sku_1", "confidence": 0.9},
        ],
    )
    assert len(res_missing["matches"]) == 1
    assert len(res_missing["missing"]) == 1
