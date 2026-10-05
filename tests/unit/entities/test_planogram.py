"""Unit tests for Planogram entity."""
from shelf_monitor.entities.common import BoundingBox
from shelf_monitor.entities.planogram import Planogram, PlanogramGrid, PlanogramItem, PlanogramRow


def test_planogram_structure():
    item0 = PlanogramItem(item_idx=0, bbox=BoundingBox(x1=10, y1=10, x2=40, y2=50), sku_id="sku_1")
    item1 = PlanogramItem(item_idx=1, bbox=BoundingBox(x1=50, y1=10, x2=80, y2=50), sku_id="sku_2")
    row0 = PlanogramRow(row_idx=0, avg_y=30.0, items=[item0, item1])

    grid = PlanogramGrid(rows=[row0])
    assert grid.total_items == 2

    plano = Planogram(shelf_id="S1", grid=grid)
    assert plano.shelf_id == "S1"
    assert plano.grid.get_cell(0, 1).sku_id == "sku_2"
    assert plano.grid.get_cell(0, 99) is None
