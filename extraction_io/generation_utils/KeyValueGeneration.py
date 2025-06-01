from pydantic import BaseModel, Field

class KeyValueGeneration(BaseModel):
    """
    Schema expected from VLM for a key-value extraction prompt.
    """
    field_name: str = Field(..., description="The logical field name, e.g. 'BorrowerName'.")
    value: str = Field(..., description="The extracted raw text for this field.")
    continue_next_page: bool = Field(
        ..., description="True if extraction should continue on the next page, else false."
    )