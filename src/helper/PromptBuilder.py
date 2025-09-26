# src/helper/prompt_builder.py

import json
from typing import Any, Dict
from common import CallableComponent
from extraction_io.ExtractionItems import ExtractionItem  # adjust import path if needed
from config.loader import prompts


class PromptBuilder(CallableComponent):
    """
    Builds prompts by loading YAML templates and embedding Pydantic JSON schemas.
    Chooses instruction fragments based on flags present in ExtractionItem.

    Whenever an instruction entry is defined in YAML as a dict with keys:
        - vars: [ …list of var names… ]
        - prompt: | … multiline template …
    this builder will attempt to fill each placeholder from either:
      1) kwargs passed into build(), or
      2) the corresponding attribute on `item`, or
      3) item.extra_rules.get(var_name),
      4) otherwise the empty string "".
    """

    _templates: Dict[str, Any] = None
    _detail: Dict[str, Any] = None

    def __init__(self):
        super().__init__()
        if PromptBuilder._templates is None:
            PromptBuilder._load_templates()

    @classmethod
    def _load_templates(cls,):
        # Ensure required top-level keys
        required = {
            "instructions_detail",
            "system",
            "postfix",
            "user",
            "fallback",
            "instructions",
        }
        missing = required - prompts.keys()
        if missing:
            raise ValueError(f"prompts.yml missing top-level keys: {missing}")

        # Validate instructions_detail
        detail = prompts["instructions_detail"]
        if not isinstance(detail, dict):
            raise ValueError("`instructions_detail` must be a dictionary")
        for group in ("boolean", "list", "option"):
            if group not in detail:
                raise ValueError(f"instructions_detail missing '{group}' section")
        if "scope" not in detail["option"]:
            raise ValueError("instructions_detail.option must contain 'scope' key")

        # All good → store both templates and detail metadata
        cls._templates = prompts
        cls._detail = detail

    def build(
        self,
        item: ExtractionItem,
        schema_dict: dict = None,
        prev_value: str = "",
        **override_vars: Any,
    ) -> str:
        """
        Build a prompt based on an ExtractionItem and an optional prev_value.
        Also accepts keyword‐args for any placeholders named in `vars: […]` for instruction templates.

        Args:
          item:          An ExtractionItem instance (with fields like search_keys, multipage_value, scope, etc.)
          schema_dict:   A dict representing the JSON schema to embed in the “system” prompt. If None, {schema} is blank.
          prev_value:    Previous concatenated text or summary (default: "").
          override_vars: Mapping of placeholder‐name → override value (highest priority).

        Returns:
          A single string combining the system‐schema prompt and the user prompt.
        """
        # 1) Determine which "user" section to use
        raw_type = item.type  # e.g. "key-value", "bullet-points", "summarization", "checkbox"
        section_key = raw_type.replace("-", "_")  # → "key_value", "bullet_points", "summarization", "checkbox"

        user_sections = list(PromptBuilder._templates["user"].keys())
        if section_key not in user_sections:
            section_key = "fallback"

        # 2) Prepare the JSON schema text for the system prompt
        schema_text = json.dumps(schema_dict, indent=2) if schema_dict else ""

        # 3) Gather instruction fragments:
        all_instr = PromptBuilder._templates["instructions"]
        generic_instr = all_instr.get("generic", {})
        type_instr = all_instr.get(section_key, {})

        # 4) Merge: generic → type-specific
        combined_instr: Dict[str, Any] = {}
        combined_instr.update(generic_instr)
        combined_instr.update(type_instr)

        # 5) Read instructions_detail so we know how to interpret each key
        detail = PromptBuilder._detail
        bool_keys = set(detail["boolean"])                 # e.g. {"multipage_value","multiline_value","single"}
        list_keys = set(detail["list"])                    # e.g. {"search_keys"}
        option_keys = set(detail["option"]["scope"])       # e.g. {"whole","section","pages","fields","single_value","multi_value"}

        instruction_parts = []

        # 5.a) If item.search_keys is nonempty, append that fragment first
        if getattr(item, "search_keys", None):
            instr_tpl = combined_instr.get("search_keys")
            if instr_tpl:
                # Format it using comma-joined list
                joined = ", ".join(item.search_keys)
                if isinstance(instr_tpl, dict):
                    rendered = self._render_from_dict(instr_tpl, item, {"search_keys": joined})
                    instruction_parts.append(rendered)
                else:
                    instruction_parts.append(instr_tpl.format(search_keys=joined))

        # --- Special handling for table.columns instruction: render columns via instruction fragment if present
        if "columns" in combined_instr:
            # Only render if table_config contains columns
            try:
                tbl_cfg = getattr(item, 'table_config', {}) or {}
                cols = tbl_cfg.get('columns', []) if isinstance(tbl_cfg, dict) else []
                if cols:
                    cols_text = "\n".join([f"    - {c}" for c in cols])
                    instr_obj = combined_instr.get('columns')
                    if isinstance(instr_obj, dict):
                        rendered_cols = self._render_from_dict(instr_obj, item, {"columns": cols_text})
                        instruction_parts.append(rendered_cols)
                    else:
                        # fallback to simple format
                        instruction_parts.append(instr_obj.format(columns=cols_text))
            except Exception:
                pass

        # 5.b) Single loop over every key in combined_instr
        for instr_key, instr_val in combined_instr.items():
            if instr_key == "search_keys" or instr_key == "columns":
                continue

            # 5.b.i) If this key is treated as a boolean‐flag
            if instr_key in bool_keys:
                if getattr(item, instr_key, False):
                    if isinstance(instr_val, dict):
                        rendered = self._render_from_dict(instr_val, item, override_vars)
                        instruction_parts.append(rendered)
                    else:
                        instruction_parts.append(instr_val)
                continue

            # 5.b.ii) If this key is one of the “option‐scopes” and equals item.scope
            if instr_key in option_keys and getattr(item, "scope", None) == instr_key:
                if isinstance(instr_val, dict):
                    rendered = self._render_from_dict(instr_val, item, override_vars)
                    instruction_parts.append(rendered)
                else:
                    instruction_parts.append(instr_val)
                continue

            # 5.b.iii) Special rule for "single" (only if type=="key-value" and
            #   both multipage_value and multiline_value are False)
            if instr_key == "single" and item.type == "key-value":
                if not getattr(item, "multipage_value", False) and not getattr(item, "multiline_value", False):
                    if isinstance(instr_val, dict):
                        rendered = self._render_from_dict(instr_val, item, override_vars)
                        instruction_parts.append(rendered)
                    else:
                        instruction_parts.append(instr_val)
                continue

            # 5.b.iv) If this is a type-specific instruction (e.g., table.header_row), append it.
            # This ensures instructions defined under the type (like table.header_row or table.merge_adjacent)
            # are included even if they aren't listed under the global boolean/option keys.
            if instr_key in type_instr:
                if isinstance(instr_val, dict):
                    rendered = self._render_from_dict(instr_val, item, override_vars)
                    instruction_parts.append(rendered)
                else:
                    instruction_parts.append(instr_val)
                continue

            # 5.b.v) Otherwise skip this instr_key

        # 5.c) If nothing got appended, fall back to generic "single" if it exists
        if not instruction_parts:
            fallback_instr = generic_instr.get("single", "")
            if fallback_instr:
                if isinstance(fallback_instr, dict):
                    rendered = self._render_from_dict(fallback_instr, item, override_vars)
                    instruction_parts.append(rendered)
                else:
                    instruction_parts.append(fallback_instr)

        instruction_text = "\n".join(instruction_parts).strip()

        # 6) Fetch and format the system + user templates
        system_tmpl = PromptBuilder._templates["system"]
        postfix_tmpl = PromptBuilder._templates["postfix"]

        if section_key in user_sections:
            user_tmpl = PromptBuilder._templates["user"][section_key]
        else:
            user_tmpl = PromptBuilder._templates["fallback"]

        # 7) Format system section (inject JSON schema)
        system_part = system_tmpl.format(
            schema=schema_text,
            field_name=item.field_name,
            description=item.description,
            extraction_type=item.type,
        )

        # 8) Format user section
        user_part = user_tmpl.format(
            field_name=item.field_name,
            description=item.description,
            extraction_type=item.type,
            prev_value=prev_value,
            prev_summary=prev_value,
            instruction=instruction_text,
            postfix=postfix_tmpl,
            search_keys=", ".join(item.search_keys),
            section_name=item.section_name or "",
            probable_pages=item.probable_pages or [],
            fields_to_summarize=item.extra.get("fields_to_summarize", []),
            columns_section="",
        )

        prev_part = ""
        if prev_value and item.multipage_value:
            prev_part = f"\n\nNote: This page is a continuation of the current, extraction value found in prev page : {prev_value}"

        # 9) Combine
        return system_part + "\n\n" + user_part + prev_part

    def _render_from_dict(
        self,
        instr_obj: Dict[str, Any],
        item: ExtractionItem,
        override_vars: Dict[str, Any],
    ) -> str:
        """
        Given an instruction entry that is a dict of the form:
            { "vars": [ var1, var2, … ],  "prompt": "… {var1} … {var2} …" }
        Attempt to fill in each placeholder by:
          1) override_vars[var_name], if present
          2) elif hasattr(item, var_name): getattr(item, var_name)
          3) elif var_name in item.extra_rules: item.extra_rules[var_name]
          4) else ""
        Returns the fully‐formatted prompt string.
        """
        vars_list = instr_obj.get("vars", [])
        template_text = instr_obj.get("prompt", "")
        subs: Dict[str, Any] = {}

        for var in vars_list:
            if var in override_vars:
                subs[var] = override_vars[var]
            elif hasattr(item, var):
                subs[var] = getattr(item, var)
            else:
                subs[var] = item.extra.get(var, "")

        # If the template references placeholders not in vars_list, fill them with ""
        try:
            return template_text.format(**subs)
        except KeyError:
            return template_text.format(**{**subs, **{k: "" for k in vars_list if k not in subs}})

    def __call__(
        self,
        item: ExtractionItem,
        schema_dict: dict = None,
        prev_value: str = "",
        **override_vars: Any,
    ) -> str:
        return self.build(item, schema_dict or {}, prev_value, **override_vars)

    def load_summary_prompts(self, prompts_path=None):
        """
        Load summary prompts from summary_prompts.yml. If prompts_path is not provided, use default location.
        """
        import os, yaml
        if prompts_path is None:
            prompts_path = os.path.join(os.path.dirname(__file__), '../../config/files/summary_prompts.yml')
        with open(prompts_path, 'r') as f:
            self._summary_prompts = yaml.safe_load(f)

    def get_summary_prompt(self, scope: str) -> str:
        """
        Get the summary prompt template for the given scope.
        Scope must be one of: extraction_items, pages, section, whole.
        """
        if not hasattr(self, '_summary_prompts'):
            self.load_summary_prompts()
        scope_map = {
            'extraction_items': 'summarise_extraction_items',
            'pages': 'summarise_pages',
            'section': 'summarise_section',
            'whole': 'summarise_whole',
        }
        key = scope_map.get(scope)
        if not key or key not in self._summary_prompts:
            raise ValueError(f"No summary prompt found for scope '{scope}'")
        return self._summary_prompts[key]['prompt']
