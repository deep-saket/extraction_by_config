from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, Field, model_validator


class TableGeneration(BaseModel):
    """Compact schema returned by the VLM prior to response formatting."""

    field_name: str = Field(..., description="Field being extracted")
    rows: List[List[str]] = Field(
        default_factory=list,
        description="Each row represented as an ordered list of cell values.",
    )
    columns: List[str] = Field(
        default_factory=list,
        description="Optional ordered list of column headers.",
    )
    continue_next_page: bool = Field(
        False,
        description="Whether additional rows for this table exist on the next page.",
    )
    page_breaks: List[int] = Field(
        default_factory=list,
        description="Optional row indices (1-based) where a new page begins.",
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_schema(cls, value: Any):
        """Allow both the compact row format and the legacy row/col schema."""
        if not isinstance(value, dict):
            return value

        raw_rows = value.get("rows") or []
        normalized_rows: List[List[str]] = []
        for row in raw_rows:
            coerced = cls._coerce_row(row)
            if coerced is not None:
                normalized_rows.append(coerced)
        value["rows"] = normalized_rows

        if not value.get("columns") and value.get("table_header"):
            header = value.get("table_header")
            if isinstance(header, list):
                value["columns"] = [cls._stringify(col) for col in header]

        return value

    @staticmethod
    def _coerce_row(row: Any) -> Optional[List[str]]:
        if row is None:
            return None
        if isinstance(row, list):
            return [TableGeneration._stringify(cell) for cell in row]
        if isinstance(row, dict):
            if "values" in row and isinstance(row["values"], list):
                source = row["values"]
            elif "cells" in row and isinstance(row["cells"], list):
                source = row["cells"]
            elif "value" in row and isinstance(row["value"], list):
                source = row["value"]
            elif "cols" in row and isinstance(row["cols"], list):
                sorted_cols = sorted(
                    row["cols"], key=lambda cell: cell.get("col_number", 0) if isinstance(cell, dict) else 0
                )
                source = [cell.get("value") if isinstance(cell, dict) else cell for cell in sorted_cols]
            else:
                return None
            return [TableGeneration._stringify(cell) for cell in source]
        # fallback to a single-value row
        return [TableGeneration._stringify(row)]

    @staticmethod
    def _stringify(value: Any) -> str:
        if value is None:
            return ""
        return str(value)
