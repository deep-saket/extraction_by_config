from common import CallableComponent

class PromptBuilder(CallableComponent):
    """
    Builds standardized prompts for extraction tasks. Depending on extraction_type,
    it will ask the VLM to return exactly one JSON object (key-value or bullet-points).
    """

    def __init__(self):
        super(PromptBuilder, self).__init__()

    def build(
        self,
        field_name: str,
        description: str,
        page_num: int,
        extraction_type: str
    ) -> str:
        """
        Returns a single‐string prompt that instructs the VLM to output only
        the JSON payload for this field. No extra text or explanation.

        Args:
          field_name:       Logical name of the field (e.g. "BorrowerName").
          description:      Human‐readable description of what to extract.
          page_num:         1‐indexed page number to look at.
          extraction_type:  Either "key-value" or "bullet-points".
        """
        if extraction_type == "key-value":
            # Prompt the VLM to return exactly one JSON object:
            return f"""
You are a document‐extraction agent. You must output exactly ONE JSON object with this schema:
{{
  "field_name": "{field_name}",
  "value": "<extracted raw text>",
  "continue_next_page": <true|false>
}}
Extract only the field "{field_name}" (description: {description}) from page {page_num}.
When you reply, DO NOT output any extra words or explanation—output only the JSON object.
"""

        elif extraction_type == "bullet-points":
            # Prompt the VLM to return exactly one JSON object for bullets:
            return f"""
You are a document‐extraction agent. You must output exactly ONE JSON object with this schema:
{{
  "field_name": "{field_name}",
  "points": ["<point1>", "<point2>", ...]
}}
Extract all bullet points for "{field_name}" (description: {description}) from page {page_num}.
When you reply, DO NOT output any extra words or explanation—output only the JSON object.
"""
        else:
            # Fallback to your old‐style prompt if a new type shows up
            return f"Extract the {field_name} ({description}) from page {page_num} as {extraction_type}."

    def __call__(
        self,
        field_name: str,
        description: str,
        page_num: int,
        extraction_type: str
    ) -> str:
        return self.build(field_name, description, page_num, extraction_type)