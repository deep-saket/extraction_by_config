# src/parsers/parse_checkbox.py

from typing import List, Dict, Any
from PIL import Image

from extraction_io.generation_utils import CheckboxGeneration
from src.parsers.ParseBase import ParseBase
from common import ExtractionState

class ParseCheckbox(ParseBase):
    """
    Concrete parser for 'checkbox' extraction. Implements:
      - _choose_schema(): Returns the CheckboxGeneration JSON schema.
      - _process_page(): Extract checkbox result from one page.
    """

    def _choose_schema(self) -> Dict[str, Any]:
        # Return the JSON schema for CheckboxGeneration
        return CheckboxGeneration.model_json_schema()

    def _process_page(
        self,
        page_num: int,
        page_result: List[Any]  # Not used for checkboxes
    ) -> Dict[str, object]:
        """
        1) Locate the image for page_num.
        2) Build and send the prompt (no prev_value).
        3) Parse the VLM output into a dict with:
           {
             "selected_option": <str> or None,
             "selected_options": <List[str]> or None,
             "continue_next_page": <bool>,
             "page_number": <int>
           }
        """
        # Find the matching image path
        image_path = None
        for (num, path) in ExtractionState.get_images():
            if num == page_num:
                image_path = path
                break

        # If no image found, return empty dict
        if image_path is None:
            return {}

        img = Image.open(image_path).convert("RGB")

        # Build the prompt (no prev_value needed)
        prompt = self.prompt_builder(self.item, self.parser_response_model_schema, "")

        # Call the VLM to get raw output
        raw_output = self.vlm_processor(img, prompt, self.parser_response_model)

        # Normalize raw_output into the expected dict
        # raw_output may be a Pydantic model or a plain dict
        if hasattr(raw_output, "selected_option") or hasattr(raw_output, "selected_options"):
            # It's a Pydantic model instance
            sel_opt = getattr(raw_output, "selected_option", None)
            sel_opts = getattr(raw_output, "selected_options", None)
            cont = getattr(raw_output, "continue_next_page", False)
        elif isinstance(raw_output, dict):
            sel_opt = raw_output.get("selected_option")
            sel_opts = raw_output.get("selected_options")
            cont = raw_output.get("continue_next_page", False)
        else:
            # Unexpected type, return empty
            return {}

        return {
            "selected_option": sel_opt,
            "selected_options": sel_opts,
            "continue_next_page": cont,
            "page_number": page_num
        }