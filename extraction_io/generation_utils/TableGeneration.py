from pydantic import BaseModel, Field
from typing import List


class TableCell(BaseModel):
    """
    A single cell in a table row.
    """
    col_number: int = Field(..., description="Column index (1-based)")
    value: str = Field(..., description="Extracted cell value")


class TableRow(BaseModel):
    """
    A single row in the extracted table.
    """
    row_number: int = Field(..., description="Row index (1-based)")
    cols: List[TableCell] = Field(..., description="List of column-value pairs in this row")


class TableGeneration(BaseModel):
    """
    Expected JSON schema from VLM for a table extraction fragment (per page or combined).
    """
    field_name: str = Field(..., description="filed_name provided")
    rows: List[TableRow] = Field(..., description="List of extracted rows")
    continue_next_page: bool = Field(..., description="Whether to continue to next page")