from pydantic import BaseModel, Field, RootModel, model_validator, PrivateAttr
from typing import List, Optional, Union, Dict, Any


# 1) Shared bounding box model for visual extractions
class BoundingBox(BaseModel):
    x1: float = Field(..., description="Left coordinate in pixels")
    y1: float = Field(..., description="Top coordinate in pixels")
    x2: float = Field(..., description="Right coordinate in pixels")
    y2: float = Field(..., description="Bottom coordinate in pixels")


# 2) Fragment model for multi-page key-value extractions
class KVFragment(BaseModel):
    value: str = Field(..., description="Raw fragment from this page")
    post_processing_value: Optional[str] = Field(
        None, description="Normalized/cleaned fragment"
    )
    page_number: int = Field(..., description="1-indexed page number")


# 3) Top-level Key-Value output model
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


# 4) Generic PointFragment model (used for bullet-points and checkbox selections)
class PointFragment(BaseModel):
    value: str = Field(..., description="Text value of this point fragment")
    post_processing_value: Optional[str] = Field(
        None, description="Normalized/cleaned text of this point fragment"
    )
    page_number: int = Field(..., description="1-indexed page number")
    index: int = Field(..., description="Global index of this point fragment")


# 5) Top-level Bullet-Points output model
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


# 6) Final summary output model
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
    def check_summary_nonempty(cls, values: "SummaryOutput") -> "SummaryOutput":
        if values.value is None:
            raise ValueError("Summary must be a non-empty string.")
        return values


# 7) Checkbox output model (now using List[PointFragment] for selected_options)
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


# 8) Table row fragment model for multi-page table support

class TableCell(BaseModel):
    """
    A single cell in a table row. Column can be addressed by index (1-based)
    and optionally by resolved column name (if headers are known).
    """
    col: int = Field(..., description="1-indexed column number")
    value: str = Field(..., description="Extracted cell value")
    col_name: Optional[str] = Field(
        None, description="Resolved column name if available (from headers)"
    )


class TableRow(BaseModel):
    """
    One logical table row (possibly aggregated across pages).
    """
    row: int = Field(..., description="1-indexed global row index")
    page_number: Optional[int] = Field(
        None, description="Source page for this row (if known)"
    )
    cells: List[TableCell] = Field(..., description="Cells for this row")


class TableRowFragment(BaseModel):
    """
    Per-page fragment (for multipage_detail), mirroring the structure of TableRow.
    Use this to keep page-wise provenance when you later concatenate rows.
    """
    index: int = Field(..., description="1-indexed row index within the overall table")
    page_number: int = Field(..., description="1-indexed page where this fragment was found")
    cells: List[TableCell] = Field(..., description="Cells captured on this page")


# 9) Table output model for structured table extraction results
class TableOutput(BaseModel):
    field_name: str = Field(..., description="Logical field name of the table")
    value: List[TableRow] = Field(..., description="Final, ordered list of table rows")
    key: str = Field(..., description="Literal search key used for extraction")
    columns: List[str] = Field(
        default_factory=list,
        description="Resolved header row (ordered list of column names). May be empty if unknown."
    )
    page_numbers: List[int] = Field(
        default_factory=list, description="Pages this table spans (unique, sorted)"
    )
    multipage_detail: Optional[List[TableRowFragment]] = Field(
        None,
        description=(
            "Optional per-page breakdown of rows/cells before aggregation. "
            "Useful for provenance/debug when tables span multiple pages."
        )
    )

    @model_validator(mode="after")
    def _validate_non_empty_rows(cls, v: "TableOutput") -> "TableOutput":
        if not v.value:
            raise ValueError("TableOutput.value must contain at least one row.")
        return v


# 10) Entity block fragment / output models
class EntityFieldEntry(BaseModel):
    key: str = Field(..., description="Logical name of the sub-field (e.g., 'phone', 'email').")
    value: str = Field(..., description="Extracted value for this sub-field.")
    confidence: Optional[float] = Field(
        None, description="Optional confidence score for this sub-field."
    )


class EntityBlockFragment(BaseModel):
    page_number: int = Field(..., description="1-indexed page number containing the fragment")
    value: str = Field(..., description="Extracted text for this fragment")
    bbox: Optional[BoundingBox] = Field(
        None, description="Optional bounding box for the fragment"
    )
    confidence: Optional[float] = Field(
        None, description="Optional confidence score for this fragment"
    )
    continue_next_page: Optional[bool] = Field(
        None, description="True if the block continues on the next page."
    )
    fields: Optional[List[EntityFieldEntry]] = Field(
        default_factory=list,
        description="Optional structured sub-fields captured within this fragment."
    )


class EntityBlockOutput(BaseModel):
    field_name: str = Field(..., description="Logical field name for the entity block")
    value: str = Field(..., description="Primary text extracted for the block")
    key: str = Field(..., description="Literal search key/description used for extraction")
    page_number: Optional[int] = Field(
        None, description="Primary page number associated with the block"
    )
    bbox: Optional[BoundingBox] = Field(
        None, description="Bounding box corresponding to the primary block"
    )
    confidence: Optional[float] = Field(
        None, description="Optional confidence score associated with the block"
    )
    fields: Optional[List[EntityFieldEntry]] = Field(
        default_factory=list,
        description="Optional structured sub-fields extracted from the block."
    )
    fragments: Optional[List[EntityBlockFragment]] = Field(
        None,
        description="Optional per-page breakdown when multipage extraction is enabled."
    )


# 11) Now define ExtractionOutput as a RootModel of the union
class ExtractionOutput(RootModel[Union[
    KeyValueOutput,
    BulletPointsOutput,
    SummaryOutput,
    CheckboxOutput,
    TableOutput,
    EntityBlockOutput
]]):
    root: Union[
        KeyValueOutput,
        BulletPointsOutput,
        SummaryOutput,
        CheckboxOutput,
        TableOutput,
        EntityBlockOutput
    ]
    # Use a private attribute for 'interim' so RootModel doesn't break; private attrs are not serialized
    _interim: bool = PrivateAttr(default=False)

# 12) And define ExtractionOutputs as a RootModel of a list of ExtractionOutput
class ExtractionOutputs(RootModel[List[ExtractionOutput]]):
    root: List[ExtractionOutput]

    def dict_by_field(self) -> dict:
        """
        Return a simple { field_name: value_or_points_or_summary_or_checkbox } mapping for quick access.
        - For KeyValueOutput: maps to the 'value' string.
        - For BulletPointsOutput: maps to the list of raw bullet fragment strings.
        - For SummaryOutput: maps to the 'summary' string.
        - For CheckboxOutput: maps to the list of PointFragment dictionaries.
        - For EntityBlockOutput: maps to a dict containing value, page_number, and bbox (if available).
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
            elif isinstance(obj, TableOutput):
                # Convert rows → dicts using headers when available; else fallback to col_1, col_2...
                header = obj.columns or []
                row_dicts: List[Dict[str, Any]] = []
                for row in obj.value:
                    rd: Dict[str, Any] = {}
                    for cell in row.cells:
                        key = (
                                cell.col_name
                                or (header[cell.col - 1] if 0 < cell.col <= len(header) else f"col_{cell.col}")
                        )
                        rd[key] = cell.value
                    row_dicts.append(rd)
                flat[obj.field_name] = row_dicts
            elif isinstance(obj, EntityBlockOutput):
                payload: Dict[str, Any] = {
                    "value": obj.value,
                    "page_number": obj.page_number
                }
                if obj.bbox is not None:
                    payload["bbox"] = obj.bbox.model_dump()
                if obj.confidence is not None:
                    payload["confidence"] = obj.confidence
                if obj.fields:
                    payload["fields"] = [field.model_dump() for field in obj.fields]
                if obj.fragments:
                    payload["fragments"] = [frag.model_dump() for frag in obj.fragments]
                flat[obj.field_name] = payload
            else:  # CheckboxOutput
                # Represent each PointFragment as its dict (point_number, value, page_number, etc.)
                flat[obj.field_name] = [pt.model_dump() for pt in obj.value]
        return flat
