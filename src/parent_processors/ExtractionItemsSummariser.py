from models import ModelManager
from common import CallableComponent
from config.loader import settings
from src.helper import LMProcessor
from  extraction_io.generation_utils import SummaryGeneration
from common import ExtractionState


class ExtractionItemsSummariser(CallableComponent):
    def __init__(self):
        super().__init__()

        parser_cfg = settings.get("parser", {}).get("args", {})
        self.vlm_candidate = parser_cfg.get("vlm_candidate")

        if not self.vlm_candidate:
            raise ValueError("VLM candidate not found in config.")

        self.lm_processor = LMProcessor(getattr(ModelManager, self.vlm_candidate))


    def __call__(self, raw_data, *args, **kwargs):
        """
        Summarize the raw_data content using ModelManager.
        """

        extraction_item = ExtractionState.get_current_extraction_item()
        processor_name = extraction_item.extra.get("parent_processor", None)
        if not processor_name:
            return raw_data

        ## get parent items
        if not extraction_item.parent:
            return raw_data

        parent_fields = []
        for parent_name in extraction_item.parent:
            parent_fields.append(ExtractionState.get_response_by_field_name(parent_name))

        summary = self.lm_processor(f"field_name = {extraction_item.field_name} || summarise {raw_data} || {parent_fields} || return in  this json format {SummaryGeneration.model_json_schema()}", SummaryGeneration)
        return summary