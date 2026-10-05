"""Unit tests for SKU entity."""
from shelf_monitor.entities.sku import SKU


def test_sku_entity():
    sku = SKU(sku_id="sku_cereal_box", name="Crunchy Cereal", category="Breakfast")
    assert sku.sku_id == "sku_cereal_box"
    assert sku.name == "Crunchy Cereal"
