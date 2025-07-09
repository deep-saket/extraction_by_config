from models import ModelManager
from common import CallableComponent, ExtractionState
from config.loader import settings
from src.helper import LMProcessor
from extraction_io.generation_utils import SummaryGeneration


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
        Summarize the raw_data content using ModelManager, including context
        from any parent fields, and return a JSON matching the SummaryOutput schema.
        """
        extraction_item = ExtractionState.get_current_extraction_item()
        # If there's no special parent_processor or no parents, just passthrough
        if not extraction_item.extra.get("parent_processor") or not extraction_item.parent:
            return raw_data

        # Gather parent-field context
        parent_contexts = []
        for parent_name in extraction_item.parent:
            resp = ExtractionState.get_response_by_field_name(parent_name)
            parent_contexts.append((parent_name, resp))

        # Build a clean prompt
        schema = SummaryGeneration.model_json_schema()
        parent_lines = "\n".join(
            f"- {name}: {resp}" for name, resp in parent_contexts
        )

        prompt = f"""
You are an expert summarizer.
Field to summarize: "{extraction_item.field_name}"

Raw content:
\"\"\"
{raw_data}
\"\"\"

Context from related fields ({len(parent_contexts)}):
{parent_lines}

Please produce a single JSON object that conforms exactly to this schema:
{schema}

Make sure your response is valid JSON with no extra commentary.
"""

        summary = self.lm_processor(prompt.strip(), SummaryGeneration)
        return summary