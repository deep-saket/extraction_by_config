# extraction_io/extraction_config_models.py

from pydantic import BaseModel, Field, RootModel, model_validator
from typing import List, Optional, Literal, Dict, Any


class ExtractionItem(BaseModel):
    field_name: str = Field(
        ...,
        description="The unique key/name of the field to extract or summarize."
    )
    description: str = Field(
        ...,
        description="A brief description of the field/section context."
    )
    probable_pages: Optional[List[int]] = Field(
        default_factory=list,
        description="(Optional) Explicit page numbers to prioritize (1-indexed)."
    )
    type: Literal["key-value", "bullet-points", "summarization", "checkbox"] = Field(
        ...,
        description="Operation type: 'key-value', 'bullet-points', 'summarization', or 'checkbox'."
    )
    multipage_value: bool = Field(
        False,
        description="(Extraction only) Whether the value may span multiple pages."
    )
    multiline_value: bool = Field(
        False,
        description="(Extraction only) Whether the value may contain multiple lines."
    )
    extra_rules: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Any additional rules. For summarization with scope='extracted_fields', "
            "use extra_rules['fields_to_summarize'] = List[str]."
        )
    )
    # Optional list of embedding-query phrases
    search_keys: Optional[List[str]] = Field(
        default_factory=list,
        description=(
            "If provided, PageFinder will embed-query each phrase to find relevant pages. "
            "Otherwise it defaults to embedding 'field_name + description'."
        )
    )

    # Single "scope" field used for both summarization and checkbox types:
    scope: Optional[
        Literal[
            "whole",
            "section",
            "pages",
            "extracted_fields",
            "single_value",
            "multi_value"
        ]
    ] = Field(
        None,
        description=(
            "When type=='summarization', valid values are:\n"
            "  • 'whole'            = entire document\n"
            "  • 'section'          = a named section (requires section_name)\n"
            "  • 'pages'            = specific pages (uses probable_pages)\n"
            "  • 'extracted_fields' = previously extracted fields (list in extra_rules['fields_to_summarize'])\n"
            "When type=='checkbox', valid values are:\n"
            "  • 'single_value'     = exactly one checkbox selected\n"
            "  • 'multi_value'      = zero or more checkboxes may be selected"
        )
    )
    section_name: Optional[str] = Field(
        None,
        description="(For scope='section', type='summarization') The heading/title of the section to locate."
    )

    @model_validator(mode="after")
    def validate_scope_for_type(cls, item: "ExtractionItem") -> "ExtractionItem":
        """
        After the model is built, ensure:
          - If type=='summarization', scope must be one of the four summarization options, and required fields exist.
          - If type=='checkbox', scope must be 'single_value' or 'multi_value'.
          - Otherwise (key-value or bullet-points) no scope is required/used.
        """
        typ = item.type
        scope = item.scope
        extra = item.extra_rules or {}

        if typ == "summarization":
            # 1) Must have a valid scope
            if scope not in ("whole", "section", "pages", "extracted_fields"):
                raise ValueError(
                    "When type=='summarization', 'scope' must be one of "
                    "['whole', 'section', 'pages', 'extracted_fields']."
                )

            # 2) If section, require section_name
            if scope == "section":
                if not item.section_name:
                    raise ValueError("When scope=='section', 'section_name' must be provided.")

            # 3) If pages, require non-empty probable_pages
            if scope == "pages":
                if not item.probable_pages:
                    raise ValueError("When scope=='pages', 'probable_pages' must be a non-empty list.")

            # 4) If extracted_fields, require extra_rules['fields_to_summarize']
            if scope == "extracted_fields":
                fields = extra.get("fields_to_summarize")
                if not fields or not isinstance(fields, list):
                    raise ValueError(
                        "When scope=='extracted_fields', extra_rules['fields_to_summarize'] "
                        "must be a non-empty list of field_name strings."
                    )

        elif typ == "checkbox":
            # For checkbox, scope must be 'single_value' or 'multi_value'
            if scope not in ("single_value", "multi_value"):
                raise ValueError(
                    "When type=='checkbox', 'scope' must be one of ['single_value', 'multi_value']."
                )

        # For 'key-value' or 'bullet-points', we do not require scope to be set.
        return item

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