from typing import List, Optional
from pydantic import BaseModel, Field


class LandingPageHints(BaseModel):
    page_index: int = Field(
        default=1,
        description="1-based index for semantic landing/start page when the first page is metadata."
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
    min_score: float = Field(
        default=0.2,
        description="Minimum score required to accept a match."
    )
    top2_margin: float = Field(
        default=0.05,
        description="Margin between top-1 and top-2 to skip tie-breakers."
    )


class ClassificationConfig(BaseModel):
    candidates: List[ClassificationCandidate] = Field(
        ...,
        description="List of candidate de_configs to select from."
    )
    thresholds: Thresholds = Field(
        default_factory=Thresholds,
        description="Threshold settings for selection."
    )
