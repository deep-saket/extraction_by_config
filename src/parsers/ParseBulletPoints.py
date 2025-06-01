# src/parsers/parse_bullet_points.py

from typing import List, Dict, Any
from PIL import Image

from extraction_io.generation_utils import BulletPointsGeneration
from src.parsers.ParseBase import ParseBase
from common import ExtractionState

class ParseBulletPoints(ParseBase):
    """
    Concrete parser for 'bullet-points' extraction. Implements:
      - _choose_schema(): Returns the BulletPoints Pydantic schema.
      - _process_page(): Extract all bullets on one page (ignoring prev_value).
    """

    def _choose_schema(self) -> Dict[str, Any]:
        # Return the JSON schema for BulletPointsOutput
        return BulletPointsGeneration.model_json_schema()

    def _process_page(
        self,
        page_num: int,
        prev_value: str  # Not used for bullet-points
    ) -> List[Dict[str, object]]:
        """
        1) Locate the image for page_num.
        2) Build and send the prompt (no prev_value used).
        3) Parse the VLM output into a list of bullet dicts.
        4) Return List[{"value": ..., "post_processing_value": None, "page_number": page_num, "point_number": idx}, ...].
        """
        # Find the matching image path
        image_path = None
        for (num, path) in ExtractionState.get_images():
            if num == page_num:
                image_path = path
                break

        # If no image found, return empty list
        if image_path is None:
            return []

        img = Image.open(image_path).convert("RGB")

        # Build the prompt (prev_value not needed)
        prompt = self.prompt_builder(self.item, self.parser_response_model_schema, "")

        # Call the VLM to get raw output
        raw_output = self.vlm_processor(img, prompt, self.parser_response_model)

        # Normalize raw_output into a list of strings
        bullets: List[str] = []
        if isinstance(raw_output, dict) and "points" in raw_output:
            bullets = raw_output["points"]
        elif hasattr(raw_output, "points"):
            bullets = getattr(raw_output, "points", [])
        elif isinstance(raw_output, list):
            bullets = [str(x) for x in raw_output]
        elif isinstance(raw_output, str):
            bullets = [line.strip() for line in raw_output.splitlines() if line.strip()]

        # Build and return the list of bullet dicts
        results: List[Dict[str, object]] = []
        idx = 1
        for b in bullets:
            results.append({
                "value": b,
                "post_processing_value": None,
                "page_number": page_num,
                "point_number": idx
            })
            idx += 1

        return results