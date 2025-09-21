from typing import List, Dict, Any, Optional
from extraction_io.ExtractionOutputs import TableOutput, TableRow, TableCell


class TableResultBuilder:
    """
    Build a unified TableOutput from per-page table fragments.

    Expected fragment shape (per page) — flexible but recommended:
      {
        "page_number": <int>,                # required
        "columns": ["Col A", "Col B", ...],  # optional (first non-empty wins)
        "rows": [
          # any of the following row shapes are accepted:
          ["valA", "valB", ...],                     # list of cell values
          {"cells": ["valA", "valB", ...]},          # explicit cells list
          {"value": ["valA", "valB", ...]},          # alt key
        ]
      }
    """

    @staticmethod
    def build(
        field_name: str,
        fragments: List[Dict[str, Any]],
        key: str,
        multipage: bool,  # accepted for API parity; not used directly
    ) -> TableOutput:
        # 1) Decide on final columns — take the first non-empty from fragments
        final_columns: List[str] = []
        for frag in fragments:
            cols = frag.get("columns") or frag.get("table_header") or []
            if isinstance(cols, list) and any(str(c).strip() for c in cols):
                final_columns = [str(c) for c in cols]
                break  # first non-empty wins

        # 2) Aggregate rows across fragments
        table_rows: List[TableRow] = []
        global_idx = 1
        page_numbers: List[int] = []

        for frag in fragments:
            page_num = int(frag.get("page_number")) if frag.get("page_number") is not None else None
            if page_num is not None:
                page_numbers.append(page_num)

            raw_rows = frag.get("rows", [])
            if not isinstance(raw_rows, list):
                continue

            for rr in raw_rows:
                # Normalize row into a list of cell values
                cells_list: Optional[List[Any]] = None
                if isinstance(rr, list):
                    cells_list = rr
                elif isinstance(rr, dict):
                    if "cells" in rr and isinstance(rr["cells"], list):
                        cells_list = rr["cells"]
                    elif "value" in rr and isinstance(rr["value"], list):
                        cells_list = rr["value"]

                if cells_list is None:
                    # Skip malformed row
                    continue

                # Build TableCell list
                cells: List[TableCell] = []
                for col_idx, cell_val in enumerate(cells_list, start=1):
                    cells.append(
                        TableCell(
                            col=col_idx,
                            value=str(cell_val) if cell_val is not None else "",
                            col_name=(final_columns[col_idx - 1] if 0 < col_idx <= len(final_columns) else None),
                        )
                    )

                table_rows.append(
                    TableRow(
                        index=global_idx,
                        page_number=page_num,
                        cells=cells,
                    )
                )
                global_idx += 1

        # 3) Build TableOutput
        return TableOutput(
            field_name=field_name,
            key=key,
            value=table_rows,
            columns=final_columns,
            page_numbers=sorted(set(p for p in page_numbers if p is not None)),
        )