from src.parsers.ParseBase import ParseBase
from extraction_io.ExtractionItems import ExtractionItem
from extraction_io.ExtractionOutputs import TableOutput
from models import ModelManager
from typing import List, Any, Optional, Dict
from extraction_io.generation_utils import TableGeneration
from common import ExtractionState


class ParseTable(ParseBase):
    """
    Uses a Vision-Language Model (VLM) to extract tables from document images/pages.
    Inherit from ParserBase for consistency with other parser types.
    """

    def _choose_schema(self) -> Dict[str, Any]:
        """
        Return the JSON schema dict for this extraction type (KeyValue or BulletPoints).
        Subclasses override this to call the appropriate Pydantic .model_json_schema().
        """
        return self.parser_response_model.model_json_schema()

    def _process_page(self, page_num: int, prev_value: str, table_header) -> Any:
        image_path = ExtractionState.get_image(page_num)
        if image_path is None:
            return None
        schema_dict = self._choose_schema()
        prompt = self.prompt_builder.build(
            self.item,
            schema_dict=schema_dict,
            prev_value=prev_value,
            table_header=table_header
        )
        gen = self.vlm_processor(image_path, prompt, TableGeneration)
        # Attach page number info to returned model for aggregation
        return { 'gen': gen, 'page': page_num }

    def run(self, pages: List[int]) -> TableOutput:
        item = self.item
        aggregated_rows: List[List[Dict[str, Any]]] = []
        header = None
        prev_value = ""

        for p in pages:
            while True:
                page_result = self._process_page(p, prev_value, header)
                if page_result is None:
                    break
                gen = page_result['gen']
                page_num = page_result['page']
                # If this page provides table_header and we haven't set header yet, use it
                if getattr(gen, 'table_header', None) and not header:
                    header = gen.table_header
                # Each row returned by the VLM may be a dict of column->value. Attach page number and index
                page_rows = []
                for idx, row in enumerate(gen.rows or []):
                    row_obj = row if isinstance(row, dict) else { 'value': str(row) }
                    row_obj['_page_number'] = page_num
                    row_obj['_index'] = len(aggregated_rows)
                    page_rows.append(row_obj)
                aggregated_rows.append(page_rows)
                # Update prev_value for context (optional)
                prev_value = "\n".join([str(r) for r in aggregated_rows])
                if not getattr(gen, 'continue_next_page', False):
                    break

        return TableOutput(key=item.field_name, value=aggregated_rows, columns=header or [], page_numbers=pages)
