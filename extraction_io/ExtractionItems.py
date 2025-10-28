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
    type: Literal[
        "key-value",
        "bullet-points",
        "summary",
        "checkbox",
        "table",
        "entity-block"
    ] = Field(
        ...,
        description="Operation type: 'key-value', 'bullet-points', 'summary', 'checkbox', 'table', or 'entity-block'."
    )
    # remove table_config; add table_header
    table_header: Optional[List[str]] = Field(
        default_factory=list,
        description="Optional list of column names/header names for table outputs. Only valid when type=='table'."
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

    anchor_phrases: Optional[List[str]] = Field(
        default_factory=list,
        description=(
            "Additional literal phrases expected near the target content. "
            "Used to bias retrieval for entity-style extractions."
        )
    )
    include_bbox: bool = Field(
        False,
        description="Request bounding-box coordinates for visual extractions (entity-block only)."
    )
    region_hint: Optional[List[str]] = Field(
        default_factory=list,
        description=(
            "Spatial hints (e.g., ['header','left']) for where the value appears. "
            "Only valid for entity-block extractions."
        )
    )
    entity_type: Optional[List[str]] = Field(
        default_factory=list,
        description="For entity-block: expected entity categories (e.g., ['company', 'person'])."
    )
    contact_fields: Optional[List[str]] = Field(
        default_factory=list,
        description="Expected sub-fields when scope targets contact information (e.g., ['name','phone'])."
    )

    # Single "scope" field used for both summary and checkbox types:
    scope: Optional[
        Literal[
            "whole",
            "section",
            "pages",
            "extraction_items",
            "single_value",
            "multi_value",
            "address_block",
            "currency",
            "identifier",
            "number",
            "date",
            "email",
            "signature_block",
            "contact",
            "person",
            "organization",
            "support",
            "billing"
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
            "  • 'multi_value'      = zero or more checkboxes may be selected\n"
            "When type in ['key-value','entity-block'], additional scopes allowed are:\n"
            "  • 'address_block', 'currency', 'identifier', 'number', 'date', 'email', 'signature_block', 'contact'.\n"
            "For contact-oriented entity blocks you may also use: 'person', 'organization', 'support', 'billing'."
        )
    )
    section_name: Optional[str] = Field(
        None,
        description="(For scope='section', type='summary') The heading/title of the section to locate."
    )

    @model_validator(mode="after")
    def validate_scope_for_type(cls, item: "ExtractionItem") -> "ExtractionItem":
        typ = item.type
        scope = item.scope
        # Enforce table-specific rules
        if typ == "table":
            # Ensure multiline_value is always True for tables
            item.multiline_value = True
            # table_header can only be present if type == table; already true here
        else:
            # If not a table, table_header must be empty
            if getattr(item, 'table_header', None):
                if item.table_header:
                    raise ValueError("'table_header' is only valid when type=='table'.")

        entity_like_types = {"entity-block"}
        contact_scopes = {"contact", "person", "organization", "support", "billing"}

        if typ in entity_like_types:
            # Visual blocks are multi-line by default
            item.multiline_value = True

            if item.anchor_phrases and not isinstance(item.anchor_phrases, list):
                raise ValueError("'anchor_phrases' must be a list when provided.")
            if item.region_hint and not isinstance(item.region_hint, list):
                raise ValueError("'region_hint' must be a list when provided.")
            if item.entity_type and not isinstance(item.entity_type, list):
                raise ValueError("'entity_type' must be a list when provided.")
            if item.contact_fields and not isinstance(item.contact_fields, list):
                raise ValueError("'contact_fields' must be a list when provided.")
            if item.contact_fields and item.scope not in contact_scopes:
                raise ValueError(
                    "'contact_fields' requires scope to be one of ['contact','person','organization','support','billing']."
                )
        else:
            if item.include_bbox:
                raise ValueError("'include_bbox' is only valid when type=='entity-block'.")
            if item.region_hint:
                raise ValueError("'region_hint' is only valid when type=='entity-block'.")
            if item.anchor_phrases:
                # Allow anchor phrases on key-value for now
                if typ != "key-value":
                    raise ValueError("'anchor_phrases' is only valid for type in ['key-value','entity-block'].")
            if item.entity_type:
                raise ValueError("'entity_type' is only valid when type=='entity-block'.")
            if item.contact_fields:
                raise ValueError("'contact_fields' is only valid when type=='entity-block'.")

        allowed_region_hints = {
            "header",
            "footer",
            "left",
            "right",
            "center",
            "top_left",
            "top_right",
            "bottom_left",
            "bottom_right"
        }
        if item.region_hint:
            invalid = [hint for hint in item.region_hint if hint not in allowed_region_hints]
            if invalid:
                raise ValueError(
                    f"'region_hint' contains unsupported values {invalid}. "
                    f"Allowed values: {sorted(allowed_region_hints)}."
                )

        if typ == "summary":
            if scope not in ("whole", "section", "pages", "extraction_items"):
                raise ValueError(
                    "When type=='summary', 'scope' must be one of ['whole', 'section', 'pages', 'extraction_items']."
                )
            # 2) If section, require section_name
            if scope == "section" and not item.section_name:
                raise ValueError("When scope=='section', 'section_name' must be provided.")
            # 3) If pages, require non-empty probable_pages
            if scope == "pages" and not item.probable_pages:
                raise ValueError("When scope=='pages', 'probable_pages' must be a non-empty list.")
            # 4) If extraction_items, require non-empty parent
            if scope == "extraction_items" and (not item.parent or not isinstance(item.parent, list)):
                raise ValueError("When scope=='extraction_items', 'parent' must be a non-empty list of field_name strings.")

            if scope in ["extraction_items", "pages"]:
                item.extra = {"parent_processor": "ExtractionItemsSummariser"}

        elif typ == "checkbox":
            if scope not in ("single_value", "multi_value"):
                raise ValueError(
                    "When type=='checkbox', 'scope' must be one of ['single_value', 'multi_value']."
                )
        elif typ in ("key-value", "entity-block"):
            allowed_scope_vals = {
                None,
                "address_block",
                "currency",
                "identifier",
                "number",
                "date",
                "email",
                "signature_block",
                "contact",
                "person",
                "organization",
                "support",
                "billing"
            }
            if scope not in allowed_scope_vals:
                raise ValueError(
                    "When type in ['key-value','entity-block'], 'scope' must be one of "
                    "['address_block','currency','identifier','number','date','email','signature_block',"
                    "'contact','person','organization','support','billing'] or omitted."
                )
        # For other types, no scope required
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
