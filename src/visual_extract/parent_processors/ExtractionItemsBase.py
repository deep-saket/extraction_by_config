import json
from typing import Any, Dict, List, Optional

from common import CallableComponent, ExtractionState
from config.loader import settings
from src.visual_extract.helper import LMProcessor
from src.visual_extract.helper.PromptBuilder import PromptBuilder
from extraction_io.generation_utils import (
    KeyValueGeneration,
    BulletPointsGeneration,
    SummaryGeneration,
    CheckboxGeneration,
    TableGeneration,
    EntityBlockGeneration,
)


class ExtractionItemsBase(CallableComponent):
    """
    Fallback parent processor:
      - Triggered when an item declares parents but no explicit parent_processor.
      - Uses LMProcessor (text-only) with parent outputs as context (no VLM call).
      - Produces fragments compatible with ResultBuilders, including page numbers
        derived from parent responses.
    """

    _model_map: Dict[str, Any] = {
        "key-value": KeyValueGeneration,
        "bullet-points": BulletPointsGeneration,
        "summary": SummaryGeneration,
        "checkbox": CheckboxGeneration,
        "table": TableGeneration,
        "entity-block": EntityBlockGeneration,
    }

    def __init__(self):
        super().__init__()
        parser_cfg = settings.get("parser", {}).get("args", {})
        lm_candidate = parser_cfg.get("vlm_candidate")
        if not lm_candidate:
            raise ValueError("LM candidate not found in config.")
        self.lm_processor = LMProcessor(getattr(__import__('models').ModelManager, lm_candidate))
        self.prompt_builder = PromptBuilder()

    def __call__(self, raw_data: Any, *args, **kwargs) -> Any:
        """
        If parents exist and no custom processor is set, use LM over parent content
        to produce fragments consumable by ResultBuilders.
        """
        extraction_item = ExtractionState.get_current_extraction_item()
        if not extraction_item or not extraction_item.parent:
            return raw_data

        parent_entries = self._collect_parent_entries(extraction_item.parent)
        if not parent_entries:
            return raw_data

        gen_model_cls = self._model_map.get(extraction_item.type)
        if not gen_model_cls:
            return raw_data

        prompt = self.prompt_builder(
            extraction_item,
            schema_dict=gen_model_cls.model_json_schema(),
            prev_value=json.dumps(parent_entries, ensure_ascii=False, indent=2)
        )

        lm_response = self.lm_processor(prompt.strip(), gen_model_cls, item=extraction_item)
        if lm_response is None:
            return raw_data

        return self._normalize_output(extraction_item.type, lm_response, parent_entries)

    def _collect_parent_entries(self, parent_names: List[str]) -> List[Dict[str, Any]]:
        entries = []
        for parent_name in parent_names:
            resp = ExtractionState.get_response_by_field_name(parent_name)
            if not resp:
                continue
            root = getattr(resp, "root", resp)
            entries.append({
                "field_name": getattr(root, "field_name", parent_name),
                "value": getattr(root, "value", ""),
                "page_number": getattr(root, "page_number", None),
            })
        return entries

    def _normalize_output(
        self,
        extype: str,
        lm_response: Any,
        parent_entries: List[Dict[str, Any]]
    ) -> Any:
        """
        Convert LM response into fragments expected by ResultBuilders.
        """
        default_page = parent_entries[0].get("page_number") or 1

        if extype == "key-value":
            val = getattr(lm_response, "value", "") if lm_response else ""
            return [{"value": val, "post_processing_value": None, "page_number": default_page}]

        if extype == "bullet-points":
            points = getattr(lm_response, "points", None) or []
            fragments = []
            for idx, p in enumerate(points):
                fragments.append({
                    "value": p,
                    "post_processing_value": None,
                    "page_number": default_page,
                    "point_number": idx
                })
            return fragments

        if extype == "checkbox":
            selected_option = getattr(lm_response, "selected_option", None)
            selected_options = getattr(lm_response, "selected_options", None)
            fragment = {
                "selected_option": selected_option,
                "selected_options": selected_options,
                "continue_next_page": False,
                "page_number": default_page
            }
            return [fragment]

        if extype == "table":
            rows = getattr(lm_response, "rows", None) or []
            columns = getattr(lm_response, "columns", None) or []
            return [{"rows": rows, "columns": columns, "page_number": default_page}]

        if extype == "entity-block":
            val = getattr(lm_response, "value", "") if lm_response else ""
            return [{"value": val, "post_processing_value": None, "page_number": default_page}]

        if extype == "summary":
            return lm_response

        return lm_response
