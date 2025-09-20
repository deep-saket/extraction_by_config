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
    type: Literal["key-value", "bullet-points", "summary", "checkbox", "table"] = Field(
        ...,
        description="Operation type: 'key-value', 'bullet-points', 'summary', 'checkbox', or 'table'."
    )
    ##TODO: Add validation that table_config is only used if type=='table'
    table_config: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Table-specific config: columns, header_row, etc. Only used if type='table'."
    )
    multipage_value: bool = Field(
        False,
        description="(Extraction only) Whether the value may span multiple pages."
    )
    multiline_value: bool = Field(
        False,
        description="(Extraction only) Whether the value may contain multiple lines."
    )
    parent: Optional[List[str]] = Field(
        default_factory=list,
        description="List of parent ExtractionItem field_names that must be processed before this item."
    )
    extra: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Any additional rules. For summary with scope='extraction_items', "
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

    # Single "scope" field used for both summary and checkbox types:
    scope: Optional[
        Literal[
            "whole",
            "section",
            "pages",
            "extraction_items",
            "single_value",
            "multi_value"
        ]
    ] = Field(
        None,
        description=(
            "When type=='summary', valid values are:\n"
            "  • 'whole'            = entire document\n"
            "  • 'section'          = a named section (requires section_name)\n"
            "  • 'pages'            = specific pages (uses probable_pages)\n"
            "  • 'extraction_items' = previously extracted fields (list in extra_rules['fields_to_summarize'])\n"
            "When type=='checkbox', valid values are:\n"
            "  • 'single_value'     = exactly one checkbox selected\n"
            "  • 'multi_value'      = zero or more checkboxes may be selected"
        )
    )
    section_name: Optional[str] = Field(
        None,
        description="(For scope='section', type='summary') The heading/title of the section to locate."
    )

    @model_validator(mode="after")
    def validate_scope_for_type(cls, item: "ExtractionItem") -> "ExtractionItem":
        """
        After the model is built, ensure:
          - If type=='summary', scope must be one of the four summary options, and required fields exist.
          - If type=='checkbox', scope must be 'single_value' or 'multi_value'.
          - Otherwise (key-value or bullet-points) no scope is required/used.
        """
        typ = item.type
        scope = item.scope
        extra = item.extra or {}

        if typ == "summary":
            # 1) Must have a valid scope
            if scope not in ("whole", "section", "pages", "extraction_items"):
                raise ValueError(
                    "When type=='summary', 'scope' must be one of "
                    "['whole', 'section', 'pages', 'extraction_items']."
                )

            # 2) If section, require section_name
            if scope == "section":
                if not item.section_name:
                    raise ValueError("When scope=='section', 'section_name' must be provided.")

            # 3) If pages, require non-empty probable_pages
            if scope == "pages":
                if not item.probable_pages:
                    raise ValueError("When scope=='pages', 'probable_pages' must be a non-empty list.")

            # 4) If extraction_items, require extra_rules['fields_to_summarize']
            if scope == "extraction_items":                
                if not item.parent or not isinstance(item.parent, list):
                    raise ValueError(
                        "When scope=='extraction_items', item.parent "
                        "must be a non-empty list of field_name strings."
                    )
                item.extra["parent_processor"] = "ExtractionItemsSummariser"

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

    def has_checkbox_items(self) -> bool:
        """
        Check if any ExtractionItem in the list has type 'checkbox'.

        Returns:
            bool: True if at least one checkbox item exists, False otherwise.
        """
        return any(item.type == "checkbox" for item in self.root)

    def sort_by_dependencies(self) -> None:

        """
        Sort extraction items based on parent dependencies.
        Items with no parents come first, followed by items whose parents are already processed.
        """
        field_to_item = {item.field_name: item for item in self.root}
        processed = set()
        result = []

        def process_item(item: ExtractionItem) -> None:
            if item.field_name in processed:
                return

            # Process parents first
            for parent in item.parent:
                if parent not in field_to_item:
                    raise ValueError(f"Parent item '{parent}' not found for '{item.field_name}'")
                if parent not in processed:
                    process_item(field_to_item[parent])

            result.append(item)
            processed.add(item.field_name)

        # Process all items
        for item in self.root:
            process_item(item)

        self.root = result
