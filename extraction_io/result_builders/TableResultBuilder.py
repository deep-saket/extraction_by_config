from extraction_io.ExtractionOutputs import TableOutput
from typing import List, Any

class TableResultBuilder:
    """
    Formats and validates table extraction results for output.
    """
    @staticmethod
    def build(field_name: str, columns: List[str], rows: List[Any], page_numbers: List[int]) -> TableOutput:
        # Add any post-processing or validation here
        return TableOutput(columns=columns, rows=rows, page_numbers=page_numbers)
