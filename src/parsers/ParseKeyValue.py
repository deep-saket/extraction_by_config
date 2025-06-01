# src/parsers/parse_key_value.py

from typing import List, Dict, Any
from PIL import Image

from extraction_io.generation_utils import KeyValueGeneration
from src.parsers.ParseBase import ParseBase
from common import ExtractionState

class ParseKeyValue(ParseBase):
    """
    Concrete parser for 'key-value' extraction. Implements:
      - _choose_schema(): Returns the KV Pydantic schema.
      - _process_page(): Extract one fragment per page, updating prev_value.
    """

    def _choose_schema(self) -> Dict[str, Any]:
        # Return the JSON schema for KeyValueOutput
        return KeyValueGeneration.model_json_schema()

    def _process_page(
        self,
        page_num: int,
        prev_value: str
    ) -> Dict[str, object]:
        """
        1) Locate the image for page_num.
        2) Build and send the prompt (including prev_value).
        3) Parse the VLM output into a single dict.
        4) Return {"value": ..., "post_processing_value": ..., "page_number": page_num}.
        """
        # Find the matching image path
        image_path = None
        for (num, path) in ExtractionState.get_images():
            if num == page_num:
                image_path = path
                break

        # If no image found (shouldn't happen if pages list is valid), return None
        if image_path is None:
            return None

        img = Image.open(image_path).convert("RGB")

        # Build the prompt using the PromptBuilder (passes previous concatenated value)
        prompt = self.prompt_builder(self.item, self.parser_response_model_schema, prev_value)

        # Call the VLM to get raw output
        raw_output = self.vlm_processor(img, prompt, self.parser_response_model)

        # Normalize raw_output into primitives
        if isinstance(raw_output, dict):
            val  = raw_output.get("value", "")
            post = raw_output.get("post_processing_value", None)
        elif hasattr(raw_output, "value"):
            val  = getattr(raw_output, "value", "")
            post = getattr(raw_output, "post_processing_value", None)
        else:
            val  = str(raw_output)
            post = None

        return {
            "value": val,
            "post_processing_value": post,
            "page_number": page_num
        }