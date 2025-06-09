# src/parsers/parse_summary.py

from typing import  Dict, Any, Optional
from PIL import Image
from extraction_io.generation_utils import SummaryGeneration
from src.parsers.ParseBase import ParseBase
from common import ExtractionState


class ParseSummary(ParseBase):
    """
    Concrete parser for 'summarization' extraction. Implements:
      - _choose_schema(): Returns the SummaryGeneration schema.
      - _process_page(): Extract one summary fragment per page, updating prev_summary.
    """

    def _choose_schema(self) -> Dict[str, Any]:
        # Return the JSON schema for SummaryGeneration
        return SummaryGeneration.model_json_schema()

    def _process_page(
        self,
        page_num: int,
        prev_summary: str
    ) -> Optional[Dict[str, Any]]:
        """
        1) Locate the image for page_num.
        2) Build and send the prompt (including prev_summary).
        3) Parse the VLM output into a SummaryGeneration instance.
        4) Return a dict with keys:
           {
             "summary": <fragment string or dict>,
             "continue_next_page": <bool>,
             "page_number": page_num
           }
        """
        # Find the matching image path
        image_path = None
        for (num, path) in ExtractionState.get_images():
            if num == page_num:
                image_path = path
                break

        # If no image found (shouldn't happen if page list is valid), skip
        if image_path is None:
            return None

        img = Image.open(image_path).convert("RGB")

        # Build the prompt using the PromptBuilder (passes previous concatenated summary)
        prompt = self.prompt_builder(self.item, prev_summary)

        # Call the VLM to get raw output
        raw_output = self.vlm_processor(img, prompt, typ="summarization")

        if isinstance(raw_output, dict):
            val = raw_output.get("value", "")
            post = raw_output.get("post_processing_value", None)
        elif hasattr(raw_output, "value"):
            val = getattr(raw_output, "value", "")
            post = getattr(raw_output, "post_processing_value", None)
        else:
            val = str(raw_output)
            post = None

        return {
            "value": val,
            "post_processing_value": post,
            "page_number": page_num
        }