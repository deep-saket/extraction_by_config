# src/parsers/parse_base.py

import json
from typing import Any, Dict, List, Optional

from common import CallableComponent, ExtractionState
from extraction_io.ExtractionItems import ExtractionItem

class ParseBase(CallableComponent):
    """
    Abstract base for per-field parsers. Responsibilities:
      1) Fetch the correct JSON schema via _choose_schema().
      2) Iterate over all specified pages.
      3) Maintain a running `prev_value` string (for multipage fields).
      4) Delegate single‐page logic to _process_page().
      5) Expose a __call__ alias so instances can be invoked directly.

    Subclasses MUST implement:
      - _choose_schema() -> Dict[str, Any]: return a Pydantic-generated JSON schema.
      - _process_page(page_num: int, prev_value: str) -> Optional[Union[dict, List[dict]]]:
          • For "key-value": return a dict {"value": str, "post_processing_value": Optional[str], "page_number": int}.
          • For "bullet-points": return a List[dict], each {"value": str, "post_processing_value": None, "page_number": int, "point_number": int}.
          • Return None if no data should be added for that page.
    """

    def __init__(
        self,
        item: ExtractionItem,
        vlm_processor: Any,      # The VLM inference engine (implements .infer(image, prompt, typ=...))
        prompt_builder: Any,      # The PromptBuilder (callable with (item, prev_value) -> str)
        parser_response_model: Any,
    ):
        super().__init__()
        self.item = item
        self.vlm_processor = vlm_processor
        self.prompt_builder = prompt_builder
        self.parser_response_model = parser_response_model
        self.parser_response_model_schema = parser_response_model.model_json_schema()

    def _choose_schema(self) -> Dict[str, Any]:
        """
        Return the JSON schema dict for this extraction type (KeyValue or BulletPoints).
        Subclasses override this to call the appropriate Pydantic .model_json_schema().
        """
        raise NotImplementedError

    def _process_page(
        self,
        page_num: int,
        prev_value: str
    ) -> Optional[Any]:
        """
        Perform one-page extraction. Must:
          1) Locate the image for page_num in ExtractionState.get_images().
          2) Build a prompt via self.prompt_builder(self.item, prev_value).
          3) Run self.vlm_processor(img, prompt, typ=<extraction_type>).
          4) Normalize the raw output into:
             - A single dict (for key-value), OR
             - A list of dicts (for bullet-points).
          5) Return None if no extraction should be recorded for this page.
        """
        raise NotImplementedError

    def run(self, pages: List[int]) -> List[Dict[str, Any]]:
        """
        Orchestrate multi-page extraction:
          a. Fetch JSON schema from _choose_schema().
          b. Iterate through `pages` in order.
          c. For each page, call _process_page(page, prev_value).
             - If result is a dict, append and update prev_value with result["value"].
             - If result is a list, extend; prev_value does not change.
          d. Return a flat list of fragment/point dicts.
        """
        # 1) Pull in the Pydantic JSON schema for instructions
        schema_dict = self._choose_schema()
        schema_text = json.dumps(schema_dict, indent=2) if schema_dict else ""
        self.logger.debug(f"[ParseBase] Schema for '{self.item.field_name}':\n{schema_text}")

        all_results: List[Dict[str, Any]] = []

        for pg in pages:
            # Delegate per-page work to _process_page
            page_result = self._process_page(pg, all_results)
            if page_result is None:
                # Skip if no content extracted on this page
                continue

            # If list, extend; if dict, append and update prev_value
            if isinstance(page_result, list):
                all_results.extend(page_result)
            else:
                all_results.append(page_result)
                # Update prev_value by concatenating the raw "value" from this page

            if not self.item.multipage_value:
                break
            elif not page_result.get("multipage_value"):
                break

        return all_results

    def __call__(self, pages: List[int]) -> List[Dict[str, Any]]:
        """
        Allow the instance itself to be called with a list of pages.
        Equivalent to invoking .run(pages).
        """
        return self.run(pages)