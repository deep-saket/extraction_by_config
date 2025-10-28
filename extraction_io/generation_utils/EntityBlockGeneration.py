from typing import Optional, List
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """
    Axis-aligned bounding box for a visual block on the page.
    Coordinates are expressed in pixel space relative to the rendered page image.
    """
    x1: float = Field(..., description="Left coordinate in pixels")
    y1: float = Field(..., description="Top coordinate in pixels")
    x2: float = Field(..., description="Right coordinate in pixels")
    y2: float = Field(..., description="Bottom coordinate in pixels")


class EntityField(BaseModel):
    key: str = Field(..., description="Name of the sub-field (e.g., 'phone', 'email').")
    value: str = Field(..., description="Extracted value for this sub-field.")
    confidence: Optional[float] = Field(
        None,
        description="Optional confidence score for this sub-field."
    )


class EntityBlockGeneration(BaseModel):
    """
    Schema expected from the VLM when extracting a cohesive visual text block.
    """
    field_name: str = Field(..., description="Logical field name being extracted.")
    value: str = Field(..., description="Extracted textual content for the block.")
    page_number: int = Field(..., description="1-indexed page containing the block.")
    continue_next_page: bool = Field(
        False,
        description="True if the block continues on the next page; otherwise False."
    )
    bbox: Optional[BoundingBox] = Field(
        None,
        description="Optional bounding box describing the location of the block."
    )
    confidence: Optional[float] = Field(
        None,
        description="Optional confidence score provided by the model."
    )
    fields: Optional[List[EntityField]] = Field(
        default_factory=list,
        description="Optional structured sub-fields captured within the block."
    )
