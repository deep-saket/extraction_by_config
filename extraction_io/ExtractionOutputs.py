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
    def validate_multipage_detail(cls, values: "KeyValueOutput") -> "KeyValueOutput":
        # If multipage_detail is provided, you could verify concatenation matches 'value'
        return values


# 3) Generic PointFragment model (used for bullet-points and checkbox selections)
class PointFragment(BaseModel):
    value: str = Field(..., description="Text value of this point fragment")
    post_processing_value: Optional[str] = Field(
        None, description="Normalized/cleaned text of this point fragment"
    )
    page_number: int = Field(..., description="1-indexed page number")
    index: int = Field(..., description="Global index of this point fragment")


# 4) Top-level Bullet-Points output model
class BulletPointsOutput(BaseModel):
    field_name: str = Field(..., description="Logical field name")
    value: List[PointFragment] = Field(
        ..., description="List of bullet point fragments (flattened across pages)"
    )
    key: str = Field(..., description="Literal search key used for extraction")

    @model_validator(mode="after")
    def check_points_nonempty(cls, values: "BulletPointsOutput") -> "BulletPointsOutput":
        if not values.value:
            raise ValueError("Bullet-points extraction must have at least one point fragment.")
        return values


# 5) Final summary output model
class SummaryOutput(BaseModel):
    field_name: str = Field(..., description="Logical field name or summary identifier")
    value: str = Field(..., description="The concatenated summary text")
    key: str = Field(..., description="Literal search key used for extraction")
    page_range: Optional[List[int]] = Field(
        None, description="If summarizing specific pages, [start_page, end_page]"
    )
    related_fields: Optional[List[str]] = Field(
        None, description="If summarizing extracted fields, list their names"
    )

    @model_validator(mode="after")
    def check_summary_nonempty(cls, values: Dict[str, Any]) -> "SummaryOutput":
        if not values.value:
            raise ValueError("Summary must be a non-empty string.")
        return values


# 6) Checkbox output model (now using List[PointFragment] for selected_options)
class CheckboxOutput(BaseModel):
    field_name: str = Field(
        ..., description="Logical field name, e.g. 'OccupancyStatus' or 'FeaturesSelected'."
    )
    value: List[PointFragment] = Field(
        ..., description="List of selected checkbox fragments (empty list if none selected)."
    )
    key: str = Field(..., description="Literal search key used for extraction")

    @model_validator(mode="after")
    def check_selected_options(cls, values: "CheckboxOutput") -> "CheckboxOutput":
        # Ensure selected_options list is always provided (it can be empty)
        if values.value is None:
            raise ValueError(
                "CheckboxOutput: 'selected_options' must be provided (use [] if no selections)."
            )
        return values


# 7) Now define ExtractionOutput as a RootModel of the union
class ExtractionOutput(RootModel[Union[KeyValueOutput, BulletPointsOutput, SummaryOutput, CheckboxOutput]]):
    root: Union[KeyValueOutput, BulletPointsOutput, SummaryOutput, CheckboxOutput]


# 8) And define ExtractionOutputs as a RootModel of a list of ExtractionOutput
class ExtractionOutputs(RootModel[List[ExtractionOutput]]):
    root: List[ExtractionOutput]

    def dict_by_field(self) -> dict:
        """
        Return a simple { field_name: value_or_points_or_summary_or_checkbox } mapping for quick access.
        - For KeyValueOutput: maps to the 'value' string.
        - For BulletPointsOutput: maps to the list of raw bullet fragment strings.
        - For SummaryOutput: maps to the 'summary' string.
        - For CheckboxOutput: maps to the list of PointFragment dictionaries.
        """
        flat: Dict[str, Any] = {}
        for entry in self.root:
            obj = entry.root
            if isinstance(obj, KeyValueOutput):
                flat[obj.field_name] = obj.value
            elif isinstance(obj, BulletPointsOutput):
                flat[obj.field_name] = [pt.value for pt in obj.value]
            elif isinstance(obj, SummaryOutput):
                flat[obj.field_name] = obj.value
            else:  # CheckboxOutput
                # Represent each PointFragment as its dict (point_number, value, page_number, etc.)
                flat[obj.field_name] = [pt.model_dump() for pt in obj.value]
        return flat