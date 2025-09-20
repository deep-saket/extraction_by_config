from src.parsers.ParseBase import ParseBase
from extraction_io.ExtractionItems import ExtractionItem
from extraction_io.ExtractionOutputs import TableOutput
from models import ModelManager
from typing import List, Any, Optional

class ParseTable(ParseBase):
    """
    Uses a Vision-Language Model (VLM) to extract tables from document images/pages.
    Inherit from ParserBase for consistency with other parser types.
    """
    def __init__(self, extraction_item: ExtractionItem, vlm_processor, prompt_builder, parser_response_model):
        super().__init__(extraction_item, vlm_processor, prompt_builder, parser_response_model)
        self.vlm = getattr(ModelManager, vlm_processor.model_name, None)
        if self.vlm is None:
            raise ValueError(f"VLM candidate '{vlm_processor.model_name}' not loaded in ModelManager.")

    def run(self, pages: List[int], parent_outputs: Optional[List[Any]] = None) -> TableOutput:
        # If parent_outputs is provided, restrict extraction to those outputs
        # Otherwise, operate on the specified pages
        images = [self.vlm_processor.pdf_processor.get_page_image(p) for p in pages]
        columns = self.extraction_item.table_config.get("columns", [])
        rows = []
        page_numbers = pages
        # TODO: Implement VLM-based table extraction here, possibly using parent_outputs
        # Example: result = self.vlm.extract_table(images, columns=columns, config=self.extraction_item.table_config, parent_outputs=parent_outputs)
        # rows = result['rows']
        return TableOutput(columns=columns, rows=rows, page_numbers=page_numbers)
