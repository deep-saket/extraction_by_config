from typing import List, Optional
from pydantic import BaseModel, Field


class LandingPageHints(BaseModel):
    """
    Optional landing-page guidance:
      - Use when the semantic start page is not page 1 (e.g., first page is metadata/cover).
      - If omitted, defaults to page 1; if provided without page_index, defaults to page 1.
    """
    page_index: Optional[int] = Field(
        default=None,
        description="Optional 1-based landing/start page when the first page is metadata."
    )
    text_hints: List[str] = Field(
        default_factory=list,
        description="Phrases likely present on the landing page."
    )
    header_hints: List[str] = Field(
        default_factory=list,
        description="Header phrases expected on the landing page."
    )
    footer_hints: List[str] = Field(
        default_factory=list,
        description="Footer phrases expected on the landing page."
    )
    consider_only: bool = Field(
        default=False,
        description="If true, classify using only the landing page."
    )


class StructureHints(BaseModel):
    """
    Soft structural cues to bonus/penalize candidates:
      - All hints are optional; they act as soft signals, not hard filters.
      - Page count range applies to the whole document.
      - Section/header/footer/landing_page hints are queried via ColPali (no OCR required).
    """
    page_count_range: Optional[List[int]] = Field(
        default=None,
        description="Optional [min, max] page count."
    )
    has_tables: Optional[bool] = Field(
        default=None,
        description="Optional hint whether tables are expected."
    )
    must_contain_sections: List[str] = Field(
        default_factory=list,
        description="Expected headings/sections (scored as text queries)."
    )
    header_hints: List[str] = Field(
        default_factory=list,
        description="Header phrases expected across pages."
    )
    footer_hints: List[str] = Field(
        default_factory=list,
        description="Footer phrases expected across pages."
    )
    landing_page: Optional[LandingPageHints] = Field(
        default=None,
        description="Hints about the semantic landing page."
    )


class ClassificationCandidate(BaseModel):
    """
    One candidate route entry. Minimal requirement is the `file` pointing to a de_config.
    Text signatures are auto-derived from that de_config (field_name, description, search_keys).
    Only add `text_hints` when you want to boost additional phrases.
    """
    file: str = Field(
        ...,
        description="de_config filename to route to."
    )
    text_hints: List[str] = Field(
        default_factory=list,
        description="Optional extra keywords/phrases to boost scoring."
    )
    structure: Optional[StructureHints] = Field(
        default=None,
        description="Optional structure hints for soft validation."
    )


class Thresholds(BaseModel):
    """
    Selection thresholds:
      - min_score: lowest acceptable score for top-1; otherwise return unknown.
      - top2_margin: if top1 - top2 < margin, optionally trigger tie-break logic.
    """
    min_score: float = Field(
        default=0.2,
        description="Minimum score required to accept a match."
    )
    top2_margin: float = Field(
        default=0.05,
        description="Margin between top-1 and top-2 to skip tie-breakers."
    )


class ClassificationConfig(BaseModel):
    """
    Top-level JSON schema for classification routing.
      - candidates: list of candidate de_configs with optional hints.
      - thresholds: global scoring thresholds.
    """
    candidates: List[ClassificationCandidate] = Field(
        ...,
        description="List of candidate de_configs to select from."
    )
    thresholds: Thresholds = Field(
        default_factory=Thresholds,
        description="Threshold settings for selection."
    )
