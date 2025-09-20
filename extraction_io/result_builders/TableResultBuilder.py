from extraction_io.ExtractionOutputs import TableOutput
from typing import List, Any

class TableResultBuilder:
    """
    Formats and validates table extraction results for output.
    """
    @staticmethod
    def build(field_name: str, columns: List[str], rows: List[Any], page_numbers: List[int]) -> TableOutput:
        # Transform rows into the new structure: list of lists of dictionaries
        formatted_rows = []
        for row in rows:
            formatted_row = [
                {"value": cell, "_page_number": row.get("_page_number", None), "_index": row.get("_index", None)}
                for cell in row
            ]
            formatted_rows.append(formatted_row)

        return TableOutput(key=field_name, value=formatted_rows, columns=columns, page_numbers=page_numbers)
