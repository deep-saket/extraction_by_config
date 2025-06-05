# src/helper/prompt_builder.py

import os
import yaml
import json
from typing import Any, Dict
from common import CallableComponent
from extraction_io.ExtractionItems import ExtractionItem  # adjust import path if needed


class PromptBuilder(CallableComponent):
    """
    Builds prompts by loading YAML templates and embedding Pydantic JSON schemas.
    Chooses instruction fragments based on flags present in ExtractionItem.
    Supports passing previous page's concatenated value via prev_value (or prev_summary).
    """

    _templates: Dict[str, Any] = None

    def __init__(self, template_path: str = "config/prompts.yml"):
        super().__init__()
        if PromptBuilder._templates is None:
            # Resolve project root if CallableComponent sets it; otherwise adjust as necessary
            full_path = os.path.join(self.project_root, template_path)
            PromptBuilder._load_templates(full_path)

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
        schema_dict: dict = None,
        prev_value: str = "",
    ) -> str:
        """
        Build a prompt based on an ExtractionItem and an optional prev_value (string).

        Args:
            item:       An ExtractionItem instance (with flags: multipage_value, multiline_value, search_keys, scope, etc.).
            schema_dict: A Python dictionary representing the JSON schema for the chosen Pydantic output model.
                         If omitted or empty, {schema} will be blank.
            prev_value: Concatenated value from previous pages (if any), else empty string.

        Returns:
            A single string combining the system schema prompt and the user prompt.
        """
        # 1) Determine which "user" section to use: map ExtractionItem.type → YAML key
        raw_type = item.type  # e.g. "key-value", "bullet-points", "summarization", "checkbox"
        section_key = raw_type.replace("-", "_")  # → "key_value", "bullet_points", "summarization", "checkbox"

        user_sections = list(PromptBuilder._templates["user"].keys())
        if section_key not in user_sections:
            section_key = "fallback"

        # 2) Prepare the JSON schema text for the "system" prompt
        schema_text = json.dumps(schema_dict, indent=2) if schema_dict else ""

        # 3) Build instruction fragments in the correct order:
        instruction_parts = []

        # 3.a) If search_keys is non-empty, always include that fragment first
        if item.search_keys:
            instr = PromptBuilder._templates["instructions"].get("search_keys", "")
            if instr:
                # Insert the actual list of keys into the fragment
                joined_keys = ", ".join(item.search_keys)
                instruction_parts.append(instr.format(search_keys=joined_keys))

        # 3.b) Depending on item.type, pick exactly one further instruction
        if item.type == "key-value":
            if item.multipage_value:
                instr = PromptBuilder._templates["instructions"].get("multipage_value", "")
                instruction_parts.append(instr)
            elif item.multiline_value:
                instr = PromptBuilder._templates["instructions"].get("multiline_value", "")
                instruction_parts.append(instr)
            else:
                instr = PromptBuilder._templates["instructions"].get("single", "")
                instruction_parts.append(instr)

        elif item.type == "bullet-points":
            # For bullet_points, typically only search_keys matters.
            # If you later add a dedicated bullet_points instruction, check for it here.
            pass

        elif item.type == "summarization":
            # Pick one of summary_whole, summary_section, summary_pages, summary_fields
            scope = item.scope  # must be one of "whole","section","pages","extracted_fields"
            key = f"summary_{scope}"
            instr = PromptBuilder._templates["instructions"].get(key, "")
            if instr:
                # Format placeholders inside the summary instruction
                formatted = instr.format(
                    section_name=item.section_name or "",
                    probable_pages=item.probable_pages or [],
                    fields_to_summarize=item.extra_rules.get("fields_to_summarize", []),
                )
                instruction_parts.append(formatted)

        elif item.type == "checkbox":
            # Pick one of checkbox_single_value or checkbox_multi_value
            scope = item.scope  # must be "single_value" or "multi_value"
            if scope == "single_value":
                instr = PromptBuilder._templates["instructions"].get("checkbox_single_value", "")
                instruction_parts.append(instr)
            else:  # "multi_value"
                instr = PromptBuilder._templates["instructions"].get("checkbox_multi_value", "")
                instruction_parts.append(instr)

        else:
            # fallback: no extra instruction aside from search_keys
            pass

        # 3.c) If, for some reason, no instruction was added at all, default to "single"
        if not instruction_parts:
            instruction_parts.append(PromptBuilder._templates["instructions"]["single"])

        instruction_text = "\n".join(instruction_parts).strip()

        # 4) Fetch the templates from YAML
        system_tmpl = PromptBuilder._templates["system"]
        postfix_tmpl = PromptBuilder._templates["postfix"]

        if section_key in user_sections:
            user_template = PromptBuilder._templates["user"][section_key]
        else:
            user_template = PromptBuilder._templates["fallback"]

        # 5) Format the system section, embedding the JSON schema
        system_part = system_tmpl.format(
            schema=schema_text,
            field_name=item.field_name,
            description=item.description,
            extraction_type=item.type,
        )

        # 6) Format the user section, filling in all placeholders:
        #    - field_name, description, extraction_type
        #    - prev_value  (for key-value or summarization)
        #    - prev_summary (if summarization template uses that)
        #    - instruction: the combined instruction_text
        #    - postfix: the common postfix
        #    - search_keys: if any, so that templates can reference {search_keys}
        #    - section_name, probable_pages, fields_to_summarize (for summarization)
        user_part = user_template.format(
            field_name=item.field_name,
            description=item.description,
            extraction_type=item.type,
            prev_value=prev_value,
            prev_summary=prev_value,  # in case the template uses {prev_summary}
            instruction=instruction_text,
            postfix=postfix_tmpl,
            search_keys=", ".join(item.search_keys),
            section_name=item.section_name or "",
            probable_pages=item.probable_pages or [],
            fields_to_summarize=item.extra_rules.get("fields_to_summarize", []),
        )

        full_prompt = system_part + "\n" + user_part
        self.logger.debug(f"Built prompt for '{item.field_name}':\n{full_prompt}")
        return full_prompt

    def __call__(
        self, item: ExtractionItem, schema_dict: dict = None, prev_value: str = ""
    ) -> str:
        return self.build(item, schema_dict or {}, prev_value)