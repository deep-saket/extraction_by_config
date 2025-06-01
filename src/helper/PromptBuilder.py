import os
import yaml
import json
from typing import Any, Dict
from common import CallableComponent
from extraction_io.ExtractionItems import ExtractionItem
from extraction_io.generation_utils import KeyValueGeneration, BulletPointsGeneration

class PromptBuilder(CallableComponent):
    """
    Builds prompts by loading YAML templates and embedding Pydantic JSON schemas.
    Chooses instruction fragments based on flags present in ExtractionItem.
    Supports passing previous page's concatenated value via prev_value.
    """
    _templates: Dict[str, Any] = None

    def __init__(self, template_path: str = "config/prompts.yml"):
        super().__init__()
        if PromptBuilder._templates is None:
            PromptBuilder._load_templates(template_path)

    @classmethod
    def _load_templates(cls, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find prompts.yml at: {path}")
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        required = {"system", "postfix", "user", "fallback", "instructions"}
        missing = required - data.keys()
        if missing:
            raise ValueError(f"prompts.yml missing top-level keys: {missing}")

        if not isinstance(data["instructions"], dict) or not data["instructions"]:
            raise ValueError("prompts.yml must have a non-empty 'instructions' section")

        cls._templates = data

    def build(
        self,
        item: ExtractionItem,
        schema_dict: dict = {},
        prev_value: str = ""
    ) -> str:
        """
        Build a prompt based on an ExtractionItem and an optional prev_value (string).

        Args:
            item:       An ExtractionItem instance (with flags: multipage_value, multiline_value).
            prev_value: Concatenated value from previous pages (if any), else empty string.

        Returns:
            A single string combining the system schema prompt and the user prompt.
        """
        sections = list(PromptBuilder._templates["user"].keys())
        # Determine which section to use from extraction_type
        etype_raw = item.type  # e.g., "key-value" or "bullet-points"
        section = etype_raw.replace("-", "_")
        if section not in sections: # this can be taken from yml as de_config_prompt_mapper
            section = "fallback"

        schema_text = json.dumps(schema_dict, indent=2) if schema_dict else ""

        # Build instruction fragment by iterating over all instruction keys
        instruction_parts = []
        for instr_key, instr_templ in PromptBuilder._templates["instructions"].items():
            if instr_key != "single" and hasattr(item, instr_key) and getattr(item, instr_key):
                instruction_parts.append(instr_templ)

        # If no flags matched, use the "single" instruction by default
        if not instruction_parts:
            instruction_parts.append(PromptBuilder._templates["instructions"]["single"])

        instruction_text = "\n".join(instruction_parts)

        # Fetch common templates
        system_templ = PromptBuilder._templates["system"]
        postfix_templ = PromptBuilder._templates["postfix"]

        # Fetch user template for this section
        if section in sections:
            user_templ = PromptBuilder._templates["user"][section]
        else:
            user_templ = PromptBuilder._templates["fallback"]

        # Format the system part (embedding schema)
        system_part = system_templ.format(
            schema=schema_text,
            field_name=item.field_name,
            description=item.description,
            extraction_type=item.type
        )

        # Format the user part (inserting field_name, description, etc.)
        user_part = user_templ.format(
            field_name=item.field_name,
            description=item.description,
            extraction_type=item.type,
            prev_value=prev_value,
            instruction=instruction_text,
            postfix=postfix_templ
        )

        full_prompt = system_part + "\n" + user_part
        self.logger.debug(f"Built prompt for '{item.field_name}':\n{full_prompt}")
        return full_prompt

    def __call__(self, item: ExtractionItem, schema_dict: dict = {}, prev_value: str = "") -> str:
        return self.build(item, schema_dict, prev_value)