from pydantic import BaseModel, Field, RootModel, model_validator
from typing import List, Optional, Union, Dict, Any


# 1) Fragment model for multi-page key-value extractions
class KVFragment(BaseModel):
    value: str = Field(..., description="Raw fragment from this page")
    post_processing_value: Optional[str] = Field(
        None, description="Normalized/cleaned fragment"
    )
    page_number: int = Field(..., description="1-indexed page number")


# 2) Top-level Key-Value output model
class KeyValueOutput(BaseModel):
    field_name: str = Field(..., description="Logical field name")
    value: str = Field(
        ..., description="Concatenated (multi-page) or single-page final value"
    )
    post_processing_value: Optional[str] = Field(
        None, description="Normalized/cleaned form of 'value'"
    )
    page_number: int = Field(
        ..., description="If multi-page: starting page; else single page."
    )
    key: str = Field(..., description="Literal search key used for extraction")
    multipage_detail: Optional[List[KVFragment]] = Field(
        None,
        description=(
            "Present only if this key was flagged multi-page. Each fragment object "
            "shows the raw & post-processed text plus the page number."
        )
    )

    @model_validator(mode="after")
    def validate_multipage_detail(cls, values):
        # If multipage_detail is provided, you could verify concatenation matches 'value'.
        # Skipping actual check here.
        return values


# 3) Fragment model for bullet-points (page-level)
class BulletPoint(BaseModel):
    value: str = Field(..., description="Raw bullet text")
    post_processing_value: Optional[str] = Field(
        None, description="Normalized/cleaned bullet text"
    )
    page_number: int = Field(..., description="1-indexed page number")
    point_number: int = Field(..., description="Global index of this bullet")


# 4) Top-level Bullet-Points output model
class BulletPointsOutput(BaseModel):
    field_name: str = Field(..., description="Logical field name")
    points: List[BulletPoint] = Field(
        ..., description="List of bullets (flattened across pages)"
    )
    key: str = Field(..., description="Literal search key used for extraction")

    @model_validator(mode="after")
    def check_points_nonempty(cls, values):
        pts = values.points
        if not pts:
            raise ValueError("Bullet-points extraction must have at least one point.")
        return values


# 5) Final summary output model
class SummaryOutput(BaseModel):
    field_name: str = Field(..., description="Logical field name or summary identifier")
    summary: str = Field(..., description="The concatenated summary text")
    page_range: Optional[List[int]] = Field(
        None, description="If summarizing specific pages, [start_page, end_page]"
    )
    related_fields: Optional[List[str]] = Field(
        None, description="If summarizing extracted fields, list their names"
    )

    @model_validator(mode="after")
    def check_summary_nonempty(cls, values: Dict[str, Any]):
        if not values.get("summary"):
            raise ValueError("Summary must be a non-empty string.")
        return values


# 6) Now define ExtractionOutput as a RootModel of the union
class ExtractionOutput(RootModel[Union[KeyValueOutput, BulletPointsOutput, SummaryOutput]]):
    root: Union[KeyValueOutput, BulletPointsOutput, SummaryOutput]


# 7) And define ExtractionOutputs as a RootModel of a list of ExtractionOutput
class ExtractionOutputs(RootModel[List[ExtractionOutput]]):
    root: List[ExtractionOutput]

    def dict_by_field(self) -> dict:
        """
        Return a simple { field_name: value_or_points_or_summary } mapping for quick access.
        - For KeyValueOutput: maps to the 'value' string.
        - For BulletPointsOutput: maps to the list of raw bullet strings.
        - For SummaryOutput: maps to the summary string.
        """
        flat = {}
        for entry in self.root:
            obj = entry.root
            if isinstance(obj, KeyValueOutput):
                flat[obj.field_name] = obj.value
            elif isinstance(obj, BulletPointsOutput):
                flat[obj.field_name] = [pt.value for pt in obj.points]
            else:  # SummaryOutput
                flat[obj.field_name] = obj.summary
        return flat