from common import CallableComponent, ExtractionState
from config.loader import settings
from src.helper import LMProcessor
from src.helper.PromptBuilder import PromptBuilder
from extraction_io.generation_utils import SummaryGeneration
import os

class ExtractionItemsSummariser(CallableComponent):
    """
    Generic parent processor for summarizing raw_data with context from parent fields.
    Uses PromptBuilder to dynamically fetch prompts and LMProcessor for summarization.
    """

    def __init__(self):
        super().__init__()
        parser_cfg = settings.get("parser", {}).get("args", {})
        self.lm_candidate = parser_cfg.get("vlm_candidate")
        if not self.lm_candidate:
            raise ValueError("LM candidate not found in config.")
        self.lm_processor = LMProcessor(getattr(__import__('models').ModelManager, self.lm_candidate))
        self.prompt_builder = PromptBuilder()  # Initialize PromptBuilder

    def __call__(self, raw_data, *args, **kwargs):
        """
        Summarize raw_data using context from parent fields.
        If no parents, return raw_data unchanged.
        """
        extraction_item = ExtractionState.get_current_extraction_item()
        if not extraction_item.parent:
            return raw_data

        # Gather parent-field context
        parent_contexts = {}
        for parent_name in extraction_item.parent:
            resp = ExtractionState.get_response_by_field_name(parent_name)
            parent_contexts[resp.root.field_name] = resp.root.value

        # Build prompt using PromptBuilder
        schema = SummaryGeneration.model_json_schema()
        prompt = self.prompt_builder.get_summary_prompt(extraction_item.scope).format(
            field_name=extraction_item.field_name,
            raw_data=raw_data,
            num_parents=len(parent_contexts),
            schema=schema,
            pages=extraction_item.probable_pages or [],
            page_text=parent_contexts
        )

        summary = self.lm_processor(prompt.strip(), SummaryGeneration)
        return summary
