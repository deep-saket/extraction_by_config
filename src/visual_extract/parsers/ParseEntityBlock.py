from typing import List, Dict, Any
from PIL import Image

from src.visual_extract.parsers.ParseBase import ParseBase
from common import ExtractionState


class ParseEntityBlock(ParseBase):
    """
    Parser for 'entity-block' extraction items.
    Extracts a cohesive block of text and optional bounding-box metadata per page.
    """

    def _choose_schema(self) -> Dict[str, Any]:
        return self.parser_response_model_schema

    def _process_page(
        self,
        page_num: int,
        page_result: List[Any]
    ) -> Dict[str, Any]:
        prev_fragments = [
            {
                "page_number": fragment.get("page_number"),
                "value": fragment.get("value")
            }
            for fragment in page_result
            if isinstance(fragment, dict)
        ]
        prev_value = f"{prev_fragments}"

        image_path = None
        for (num, path) in ExtractionState.get_images():
            if num == page_num:
                image_path = path
                break

        if image_path is None:
            return None

        img = Image.open(image_path).convert("RGB")

        prompt = self.prompt_builder(
            self.item,
            self.parser_response_model_schema,
            prev_value,
        )

        raw_output = self.vlm_processor(img, prompt, self.parser_response_model)

        if hasattr(raw_output, "model_dump"):
            data = raw_output.model_dump()
        elif isinstance(raw_output, dict):
            data = raw_output
        else:
            # Fallback: treat as plain string
            data = {
                "value": str(raw_output),
                "page_number": page_num,
                "continue_next_page": False
            }

        value = data.get("value", "")
        page = data.get("page_number") or page_num
        bbox = data.get("bbox")
        confidence = data.get("confidence")
        continue_next = data.get("continue_next_page", False)
        fields = data.get("fields", [])

        return {
            "value": value,
            "page_number": page,
            "bbox": bbox,
            "confidence": confidence,
            "continue_next_page": continue_next,
            "multipage_value": continue_next,
            "fields": fields
        }
