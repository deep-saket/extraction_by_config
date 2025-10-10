from __future__ import annotations

import json
from typing import Dict, Iterable, List, Optional

from common import BaseComponent
from config.loader import auto_config_prompts


class AutoConfigPromptBuilder(BaseComponent):
    """
    Lightweight prompt builder for the auto-config workflow. It loads dedicated prompt templates
    and injects contextual information (document metadata, previously detected fields, schema).
    """

    def __init__(self):
        super().__init__()
        if not auto_config_prompts:
            raise ValueError(
                "auto_config_prompts.yml could not be loaded. Ensure the file exists and is readable."
            )

        discovery = auto_config_prompts.get("discovery") or {}
        item = auto_config_prompts.get("item") or {}
        if not discovery or not item:
            raise ValueError("auto_config_prompts.yml must define both 'discovery' and 'item' sections.")

        self._discovery_prompts = {
            "system": discovery.get("system", ""),
            "user": discovery.get("user", ""),
        }
        self._item_prompts = {
            "system": item.get("system", ""),
            "user": item.get("user", ""),
        }

    def build_discovery_prompt(
        self,
        *,
        document_name: str,
        total_pages: int,
        page_number: int,
        schema_dict: Dict,
        known_fields: Optional[Iterable[Dict]] = None,
    ) -> str:
        """
        Build the full discovery prompt (system + user) for a given page.
        """
        known_fields_block = self._render_known_fields(known_fields or [], include_type=True)
        schema_text = json.dumps(schema_dict or {}, indent=2, ensure_ascii=False)

        user_prompt = self._discovery_prompts["user"].format(
            document_name=document_name,
            page_number=page_number,
            total_pages=total_pages,
            known_fields=known_fields_block,
            schema=schema_text,
        )

        if self._discovery_prompts["system"]:
            return f"{self._discovery_prompts['system'].strip()}\n\n{user_prompt.strip()}"
        return user_prompt.strip()

    def build_item_prompt(
        self,
        *,
        document_name: str,
        total_pages: int,
        field_name: str,
        field_type: str,
        probable_pages: List[int],
        multipage_hint: Optional[bool],
        candidate_notes: Optional[str],
        schema_dict: Dict,
        known_fields: Optional[Iterable[Dict]] = None,
    ) -> str:
        """
        Build the prompt used for generating a single ExtractionItem entry.
        """
        known_fields_block = self._render_known_fields(known_fields or [], include_type=True, include_pages=True)
        schema_text = json.dumps(schema_dict or {}, indent=2, ensure_ascii=False)

        user_prompt = self._item_prompts["user"].format(
            document_name=document_name,
            total_pages=total_pages,
            field_name=field_name,
            field_type=field_type,
            probable_pages=", ".join(str(p) for p in sorted(probable_pages)) if probable_pages else "[]",
            multipage_hint=str(bool(multipage_hint)).lower(),
            candidate_notes=candidate_notes or "None",
            known_fields=known_fields_block,
            schema=schema_text,
        )

        if self._item_prompts["system"]:
            return f"{self._item_prompts['system'].strip()}\n\n{user_prompt.strip()}"
        return user_prompt.strip()

    def _render_known_fields(
        self,
        known_fields: Iterable[Dict],
        *,
        include_type: bool = False,
        include_pages: bool = False,
    ) -> str:
        """
        Format the running list of known fields into a compact bullet list for the prompt.
        """
        rendered_lines: List[str] = []
        for entry in known_fields:
            name = entry.get("field_name", "").strip() or "UnnamedField"
            parts: List[str] = []
            if include_type and entry.get("type"):
                parts.append(f"type={entry.get('type')}")
            if include_pages:
                pages = entry.get("probable_pages") or []
                if pages:
                    parts.append("pages=" + ", ".join(str(p) for p in sorted({int(p) for p in pages if isinstance(p, (int, float))})))
            if entry.get("multipage_value"):
                parts.append("multipage")
            if entry.get("notes"):
                parts.append(f"notes={entry.get('notes')}")
            suffix = f" ({'; '.join(parts)})" if parts else ""
            rendered_lines.append(f"- {name}{suffix}")

        if not rendered_lines:
            return "- None so far."

        return "\n  ".join(rendered_lines[:12])
