# extraction_io/extraction_config_models.py

from pydantic import BaseModel, Field, RootModel
from typing import List, Optional, Literal, Dict, Any

class ExtractionItem(BaseModel):
    field_name: str = Field(..., description="The unique key/name of the field to extract.")
    description: str = Field(..., description="A brief description of the field/document context.")
    probable_pages: Optional[List[int]] = Field(
        default_factory=list,
        description="Specific page numbers to prioritize for extraction (1-indexed)."
    )
    type: Literal["key-value", "bullet-points"] = Field(
        ..., description="Type of extraction: 'key-value' or 'bullet-points'."
    )
    multipage_value: bool = Field(
        False,
        description="Indicates if the extracted value may span multiple pages."
    )
    multiline_value: bool = Field(
        False,
        description="Indicates if the extracted value may contain multiple lines for a single field."
    )
    extra_rules: Dict[str, Any] = Field(
        default_factory=dict,
        description="Any additional extraction rules not explicitly defined."
    )

    class Config:
        extra = "allow"

class ExtractionItems(RootModel[List[ExtractionItem]]):
    """
    RootModel whose entire payload is a list of ExtractionItem objects.
    """
    root: List[ExtractionItem]

    def __iter__(self):
        return iter(self.root)

    def __getitem__(self, idx: int) -> ExtractionItem:
        return self.root[idx]