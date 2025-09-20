from common import CallableComponent, ExtractionState
from config.loader import settings
from src.helper import LMProcessor
from extraction_io.generation_utils import SummaryGeneration
import yaml
import os

##TODO: may be create a class for reading prompts or may be edit the existing class for reading prompts


class ExtractionItemsSummariser(CallableComponent):
    """
    Generic parent processor for summarizing raw_data with context from parent fields.
    Loads prompts from summary_prompts.yml and uses LMProcessor (not VLMs).
    """

    def __init__(self):
        super().__init__()
        parser_cfg = settings.get("parser", {}).get("args", {})
        self.lm_candidate = parser_cfg.get("vlm_candidate")
        if not self.lm_candidate:
            raise ValueError("LM candidate not found in config.")
        self.lm_processor = LMProcessor(getattr(__import__('models').ModelManager, self.lm_candidate))
        # Load summary prompts
        prompts_path = os.path.join(os.path.dirname(__file__), '../../config/files/summary_prompts.yml')
        with open(prompts_path, 'r') as f:
            self.prompts = yaml.safe_load(f)

    def __call__(self, raw_data, *args, **kwargs):
        extraction_item = ExtractionState.get_current_extraction_item()
        if not extraction_item.parent:
            return raw_data
        # Gather parent-field context
        parent_contexts = []
        for parent_name in extraction_item.parent:
            resp = ExtractionState.get_response_by_field_name(parent_name)
            parent_contexts.append((parent_name, resp))
        # Build prompt using summary_prompts.yml
        schema = SummaryGeneration.model_json_schema()
        parent_lines = "\n".join(f"- {name}: {resp}" for name, resp in parent_contexts)
        prompt_template = self.prompts['summarise_extraction_items']['prompt']
        prompt = prompt_template.format(
            field_name=extraction_item.field_name,
            raw_data=raw_data,
            num_parents=len(parent_contexts),
            parent_lines=parent_lines,
            schema=schema
        )
        summary = self.lm_processor(prompt.strip(), SummaryGeneration)
        return summary