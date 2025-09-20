from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class TableGeneration(BaseModel):
    """
    Expected JSON schema from VLM for a table extraction fragment (per page or combined):
    {
      "field_name": "<string>",
      "table_header": ["col1", "col2", ...]    # optional, present on first page if header exists
      "rows": [ {<col>: <value>, ...}, ... ],
      "continue_next_page": true|false
    }
    """
    field_name: str = Field(..., description="Logical field name")
    table_header: Optional[List[str]] = Field(None, description="Optional header row (ordered list of column names)")
    rows: List[Dict[str, Any]] = Field(..., description="List of row objects mapping column name -> value")
    continue_next_page: bool = Field(..., description="Whether to continue to next page")

