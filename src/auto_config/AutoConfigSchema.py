from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


FieldType = Literal["key-value", "bullet-points", "table", "summary", "checkbox"]


class FieldDiscoveryEntry(BaseModel):
    """
    Minimal candidate representation returned during the discovery pass.
    Only `field_name` and `type` are required; extra hints help consolidate pages.
    """

    field_name: str = Field(
        ...,
        description="Suggested unique identifier for the field (CamelCase recommended).",
    )
    type: FieldType = Field(
        ...,
        description="Extraction type the downstream parser should use.",
    )
    probable_pages: List[int] = Field(
        default_factory=list,
        description="Optional list of 1-indexed pages where the field appears.",
    )
    multipage_hint: Optional[bool] = Field(
        default=None,
        description="Set to true if the field clearly spans multiple pages.",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Optional free-form observations to aid downstream prompts.",
    )

    @model_validator(mode="after")
    def _normalise(cls, value: "FieldDiscoveryEntry") -> "FieldDiscoveryEntry":
        value.field_name = (value.field_name or "").strip()
        if " " in value.field_name:
            value.field_name = value.field_name.replace(" ", "")

        unique_pages = sorted({int(p) for p in value.probable_pages if isinstance(p, int) and p > 0})
        value.probable_pages = unique_pages

        if value.notes:
            value.notes = value.notes.strip()

        return value


class FieldDiscoveryPagePlan(BaseModel):
    """
    Structured response for a single PDF page describing candidate fields.
    """

    page_number: int = Field(..., description="1-indexed page number the response refers to.")
    page_summary: Optional[str] = Field(
        default=None,
        description="Optional short description of the page (for debugging / UX only).",
    )
    fields: List[FieldDiscoveryEntry] = Field(
        default_factory=list,
        description="List of candidate field summaries detected on this page.",
    )

    @model_validator(mode="after")
    def _adjust(cls, value: "FieldDiscoveryPagePlan") -> "FieldDiscoveryPagePlan":
        value.page_number = int(value.page_number)
        if value.page_number <= 0:
            value.page_number = max(1, value.page_number)
        return value


def extract_field_summary(entry: FieldDiscoveryEntry) -> Dict[str, Optional[str]]:
    """
    Helper utility for summarising a discovery entry in prompts or UI contexts.
    """
    return {
        "field_name": entry.field_name,
        "type": entry.type,
        "notes": entry.notes,
    }
