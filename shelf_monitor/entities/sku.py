"""SKU entity."""
from pydantic import BaseModel, Field


class SKU(BaseModel):
    """A Stock-Keeping Unit (product) known to the system."""
    sku_id: str = Field(..., description="Unique SKU identifier, e.g. 'sku_42'")
    name: str = Field(default="", description="Human-readable product name")
    category: str = Field(default="", description="Product category")
    barcode: str = Field(default="", description="EAN/UPC barcode")
    image_path: str = Field(default="", description="Path to reference product image")

    model_config = {"frozen": True}
