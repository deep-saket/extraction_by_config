from pydantic import BaseModel, Field
from typing import List


class BulletPointsGeneration(BaseModel):
    """
    Schema expected from VLM for a bullet-points extraction prompt.
    """
    field_name: str = Field(..., description="The logical field name, e.g. 'benefits_list'.")
    points: List[str] = Field(..., description="List of extracted bullet strings.")
    continue_next_page: bool = Field(
        ...,
        description="True if there is another page to process for this field; otherwise false."
    )