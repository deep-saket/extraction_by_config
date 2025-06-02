from pydantic import BaseModel, Field
from typing import Union, Dict


class SummaryGeneration(BaseModel):
    """
    Schema for summarization fragments from the VLM (one page or text-chunk at a time).
    The "summary" can be either:
      • A plain string: just the summary text for this page/fragment.
      • A dict with keys "section" and "summary": if the fragment is labeled by section.

    Expected JSON from VLM:
    {
      "field_name": "<string>",
      "summary": "<string summary>"
          OR
      "summary": { "section": "<section_name>", "summary": "<string summary>" },
      "continue_next_page": <true|false>
    }
    """
    field_name: str = Field(..., description="Logical field name or summary key")
    summary: Union[
        str,
        Dict[str, str]
    ] = Field(
        ...,
        description=(
            "Either a plain-string summary of this page/chunk, or "
            "a dict with keys:\n"
            '  • "section": name of the section this fragment belongs to\n'
            '  • "summary": the actual summary text for that section/page'
        )
    )
    continue_next_page: bool = Field(
        ..., description="Whether to continue summarizing on the next page/fragment"
    )