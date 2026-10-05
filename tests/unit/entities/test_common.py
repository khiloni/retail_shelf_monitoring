"""Unit tests for common entities."""
from shelf_monitor.entities.common import AlertType, BoundingBox, CellState


def test_bounding_box_properties():
    box = BoundingBox(x1=10.0, y1=20.0, x2=50.0, y2=80.0)
    assert box.width == 40.0
    assert box.height == 60.0
    assert box.area == 2400.0
    assert box.center == (30.0, 50.0)


def test_bounding_box_negative_clamp():
    box = BoundingBox(x1=-5.0, y1=-10.0, x2=20.0, y2=30.0)
    assert box.x1 == 0.0
    assert box.y1 == 0.0


def test_cell_states():
    assert CellState.OK == "ok"
    assert CellState.EMPTY == "empty"
    assert CellState.MISPLACED == "misplaced"
    assert CellState.UNKNOWN == "unknown"


def test_alert_types():
    assert AlertType.OOS == "out_of_stock"
    assert AlertType.MISPLACED == "misplaced"
