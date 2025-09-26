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

    def _process_page(self, page_num: int, prev_value) -> Any:
        image_path = ExtractionState.get_image(page_num)
        if image_path is None:
            return None
        schema_dict = self._choose_schema()
        prompt = self.prompt_builder.build(
            self.item,
            schema_dict=schema_dict,
            prev_value=prev_value,
        )
        gen = self.vlm_processor(image_path, prompt, TableGeneration)
        # Attach page number info to returned model for aggregation
        return { 'gen': gen, 'page': page_num }

    def _get_prev_page_context(self, prev_page_res):
        if prev_page_res:
            if prev_page_res.rows:
                return prev_page_res.rows[0]

    def run(self, pages: List[int]) -> TableOutput:
        item = self.item
        prev_value = ""
        results = []

        for pg in pages:
            start_page_res = []
            while True:
                page_result = self._process_page(pg, prev_value)
                if page_result is None or page_result['gen'] is None:
                    break
                gen = page_result['gen']
                page_num = page_result['page']

                if gen:
                    prev_value = self._get_prev_page_context(gen)
                    start_page_res.extend(gen)
                if gen and not getattr(gen, 'continue_next_page', False):
                    break
                pg += 1
            results.append(start_page_res)

        if len(results) == 1:
            results = results[0]

        return results
