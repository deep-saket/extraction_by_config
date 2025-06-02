# extraction_io/extraction_config_models.py

from pydantic import BaseModel, Field, RootModel, model_validator
from typing import List, Optional, Literal, Dict, Any


class ExtractionItem(BaseModel):
    """
    Pydantic model for a single extraction or summarization configuration item.
    """
    field_name: str = Field(..., description="The unique key/name of the field to extract or summarize.")
    description: str = Field(..., description="A brief description of the field or summarization context.")
    probable_pages: Optional[List[int]] = Field(
        default_factory=list,
        description="Specific page numbers to prioritize (1-indexed). For summarization with scope='pages', "
                    "these pages will be concatenated."
    )
    type: Literal["key-value", "bullet-points", "summarization"] = Field(
        ...,
        description="Type of operation: 'key-value', 'bullet-points', or 'summarization'."
    )
    multipage_value: bool = Field(
        False,
        description="(Extraction) Indicates if the extracted value may span multiple pages. Ignored for summarization."
    )
    multiline_value: bool = Field(
        False,
        description="(Extraction) Indicates if the extracted value may contain multiple lines. Ignored for summarization."
    )
    extra_rules: Dict[str, Any] = Field(
        default_factory=dict,
        description="Any additional rules not explicitly defined. "
                    "For summarization with scope='extracted_fields', supply "
                    "'fields_to_summarize': List[str] here."
    )

    # Summarization-specific fields (only meaningful when type=='summarization')
    summary_scope: Optional[Literal["whole", "section", "pages", "extracted_fields"]] = Field(
        None,
        description=(
            "When type=='summarization', defines what to summarize:\n"
            "  • 'whole'            = entire document\n"
            "  • 'section'          = a named section (requires section_name)\n"
            "  • 'pages'            = specific pages (uses probable_pages)\n"
            "  • 'extracted_fields' = previously extracted fields "
            "(list in extra_rules['fields_to_summarize'])"
        )
    )
    section_name: Optional[str] = Field(
        None,
        description="(For summary_scope='section') The heading/title of the section to locate."
    )

    @model_validator(mode="after")
    def validate_summarization_fields(cls, values: dict) -> dict:
        """
        Ensure that when type=='summarization', required fields are present based on summary_scope.
        """
        typ = values.type
        scope = values.summary_scope
        extra = values.extra_rules

        if typ == "summarization":
            if scope is None:
                raise ValueError("When type=='summarization', 'summary_scope' must be provided.")
            if scope == "section" and not values.section_name:
                raise ValueError("When summary_scope=='section', 'section_name' must be provided.")
            if scope == "pages":
                pages = values.probable_pages or []
                if not pages:
                    raise ValueError("When summary_scope=='pages', 'probable_pages' must be a non-empty list.")
            if scope == "extracted_fields":
                fields = extra.get("fields_to_summarize")
                if not fields or not isinstance(fields, list):
                    raise ValueError("When summary_scope=='extracted_fields', extra_rules['fields_to_summarize'] "
                                     "must be a non-empty list of field_name strings.")
        return values

    class Config:
        extra = "allow"  # permit additional keys under extra_rules for future extensions


class ExtractionItems(RootModel[List[ExtractionItem]]):
    """
    RootModel whose entire payload is a list of ExtractionItem objects.
    """
    root: List[ExtractionItem]

    def __iter__(self):
        return iter(self.root)

    def __getitem__(self, idx: int) -> ExtractionItem:
        return self.root[idx]