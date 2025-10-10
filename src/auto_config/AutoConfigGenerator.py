from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from pydantic import ValidationError

from common import BaseComponent, DirtyJsonParser
from config.loader import settings
from extraction_io.ExtractionItems import ExtractionItem, ExtractionItems
from models import ModelManager
from src.auto_config.AutoConfigPromptBuilder import AutoConfigPromptBuilder
from src.auto_config.AutoConfigSchema import FieldDiscoveryEntry, FieldDiscoveryPagePlan
from vector_retrieve import PDFProcessor


_TYPE_PRIORITY: Dict[str, int] = {
    "table": 5,
    "bullet-points": 4,
    "summary": 3,
    "checkbox": 2,
    "key-value": 1,
}


class _FieldCandidate:
    """
    Aggregates discovery hints for a single field across pages.
    """

    def __init__(self, field_name: str, field_type: str):
        self.field_name = field_name
        self.type = field_type
        self._pages: set[int] = set()
        self._samples: Dict[int, str] = {}
        self.notes: List[str] = []
        self.multipage_hint = False

    def register(self, entry: FieldDiscoveryEntry, page_number: int, image_path: str) -> None:
        self.type = self._resolve_type(self.type, entry.type)

        page_candidates = set(entry.probable_pages or [])
        page_candidates.add(page_number)
        for page in page_candidates:
            if isinstance(page, int) and page > 0:
                self._pages.add(int(page))

        if page_number not in self._samples and image_path:
            self._samples[page_number] = image_path

        self.multipage_hint = self.multipage_hint or bool(entry.multipage_hint) or len(self._pages) > 1

        if entry.notes:
            note = entry.notes.strip()
            if note and note not in self.notes:
                self.notes.append(note)

    @property
    def probable_pages(self) -> List[int]:
        return sorted(self._pages)

    @property
    def notes_text(self) -> Optional[str]:
        if not self.notes:
            return None
        return "; ".join(self.notes)

    def has_multipage(self) -> bool:
        return self.multipage_hint or len(self._pages) > 1

    def primary_image(self) -> Optional[Tuple[int, str]]:
        if not self._samples:
            return None
        page = sorted(self._samples.keys())[0]
        return page, self._samples[page]

    def to_prompt_dict(self) -> Dict[str, object]:
        return {
            "field_name": self.field_name,
            "type": self.type,
            "probable_pages": self.probable_pages,
            "multipage_value": self.has_multipage(),
            "notes": self.notes_text,
        }

    @staticmethod
    def _resolve_type(existing_type: str, new_type: str) -> str:
        existing_score = _TYPE_PRIORITY.get(existing_type, 0)
        new_score = _TYPE_PRIORITY.get(new_type, 0)
        if new_score > existing_score:
            return new_type
        return existing_type


class _FieldRegistry:
    """
    Maintains discovery candidates and preserves insertion order.
    """

    def __init__(self):
        self._entries: Dict[str, _FieldCandidate] = {}
        self._order: List[str] = []

    @staticmethod
    def _normalise(name: str) -> str:
        cleaned = (name or "").strip()
        return cleaned.replace(" ", "") or "UnnamedField"

    def register(self, entry: FieldDiscoveryEntry, page_number: int, image_path: str) -> None:
        key = self._normalise(entry.field_name)
        candidate = self._entries.get(key)
        if candidate is None:
            candidate = _FieldCandidate(key, entry.type)
            self._entries[key] = candidate
            self._order.append(key)
        candidate.register(entry, page_number, image_path)

    def prompt_summaries(self, exclude: Optional[str] = None) -> List[Dict[str, object]]:
        summaries: List[Dict[str, object]] = []
        for key in self._order:
            if exclude and key == exclude:
                continue
            summaries.append(self._entries[key].to_prompt_dict())
        return summaries

    def iter_candidates(self) -> Iterable[_FieldCandidate]:
        for key in self._order:
            yield self._entries[key]


class AutoConfigGenerator(BaseComponent):
    """
    Generates a draft extraction configuration by running a two-step VLM workflow:
      1) Discovery pass to list field candidates.
      2) Detail pass to materialise each candidate as a validated ExtractionItem entry.
    """

    def __init__(self):
        auto_config_cfg = settings.get("auto_config", {}).get("args", {})
        device_key = auto_config_cfg.get("device", "cpu")
        self.device = torch.device(device_key) if isinstance(device_key, str) else device_key

        self.vlm_candidate = auto_config_cfg.get("vlm_candidate", "QwenV25Infer")
        configured_models = list(auto_config_cfg.get("models") or [])
        if self.vlm_candidate not in configured_models:
            configured_models.append(self.vlm_candidate)
        ModelManager.initialize_models(self.device, model_classes=configured_models)
        try:
            self.vlm_infer = getattr(ModelManager, self.vlm_candidate)
        except AttributeError as exc:
            raise RuntimeError(f"ModelManager is missing the VLM candidate '{self.vlm_candidate}'.") from exc

        super().__init__(auto_config_cfg)
        self.prompt_builder = AutoConfigPromptBuilder()
        self.pdf_processor = PDFProcessor(colpali_infer=None, checkbox_infer=None, override=False)
        self.per_page_field_limit = auto_config_cfg.get("per_page_field_limit")
        self.configured_max_pages = auto_config_cfg.get("max_pages")

        self._discovery_schema = FieldDiscoveryPagePlan.model_json_schema()
        self._extraction_item_schema = ExtractionItem.model_json_schema()

    def generate(
        self,
        pdf_path: str,
        *,
        output_config_path: Optional[str] = None,
        max_pages: Optional[int] = None,
    ) -> List[dict]:
        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.is_file():
            raise FileNotFoundError(f"PDF not found at path: {pdf_path}")

        page_images = self._prepare_page_images(str(pdf_path_obj))
        total_pages = len(page_images)
        if total_pages == 0:
            raise ValueError(f"No pages found while processing: {pdf_path}")

        target_max_pages = self._resolve_max_pages(total_pages, max_pages)
        document_name = pdf_path_obj.stem
        registry = _FieldRegistry()

        for idx, (page_number, image_path) in enumerate(page_images, start=1):
            if idx > target_max_pages:
                break

            known_fields = registry.prompt_summaries()
            prompt = self.prompt_builder.build_discovery_prompt(
                document_name=document_name,
                total_pages=total_pages,
                page_number=page_number,
                schema_dict=self._discovery_schema,
                known_fields=known_fields,
            )

            plan = self._run_discovery(image_path, prompt, page_number)
            if not plan:
                continue

            fields = self._limit_fields(plan.fields)
            for entry in fields:
                registry.register(entry, page_number, image_path)

        candidates = list(registry.iter_candidates())
        if not candidates:
            return []

        extraction_items: List[dict] = []
        for candidate in candidates:
            known_fields = self._format_completed_items(extraction_items)
            prompt = self.prompt_builder.build_item_prompt(
                document_name=document_name,
                total_pages=total_pages,
                field_name=candidate.field_name,
                field_type=candidate.type,
                probable_pages=candidate.probable_pages,
                multipage_hint=candidate.has_multipage(),
                candidate_notes=candidate.notes_text,
                schema_dict=self._extraction_item_schema,
                known_fields=known_fields,
            )

            sample = candidate.primary_image()
            if sample is None:
                self.logger.warning(
                    "[AutoConfig] No sample image recorded for field '%s'. Skipping.", candidate.field_name
                )
                continue

            _, image_path = sample
            item_dict = self._generate_item_for_candidate(candidate, prompt, image_path)
            if item_dict is None:
                self.logger.warning(
                    "[AutoConfig] Failed to materialise ExtractionItem for field '%s'.", candidate.field_name
                )
                continue

            extraction_items.append(item_dict)

        if not extraction_items:
            return []

        validated = ExtractionItems.model_validate(extraction_items)
        final_items = [item.model_dump() for item in validated.root]

        if output_config_path:
            self._write_output(final_items, output_config_path)

        return final_items

    def _prepare_page_images(self, pdf_path: str) -> List[Tuple[int, str]]:
        return self.pdf_processor.pdf_to_images(pdf_path)

    def _resolve_max_pages(self, total_pages: int, override: Optional[int]) -> int:
        if override is not None:
            return min(total_pages, int(override))
        if self.configured_max_pages:
            return min(total_pages, int(self.configured_max_pages))
        return total_pages

    def _run_discovery(
        self,
        image_path: str,
        prompt: str,
        page_number: int,
    ) -> Optional[FieldDiscoveryPagePlan]:
        attempts = 2
        augmented_prompt = prompt
        last_error: Optional[Exception] = None

        for attempt in range(1, attempts + 1):
            self.logger.info(
                "[AutoConfig] Discovery VLM inference on page %s (attempt %s/%s)",
                page_number,
                attempt,
                attempts,
            )
            raw_output = self.vlm_infer.infer(image_path, augmented_prompt)

            try:
                parsed = DirtyJsonParser.parse(raw_output)
                plan = FieldDiscoveryPagePlan.model_validate(parsed)
                if plan.page_number != page_number:
                    plan.page_number = page_number
                return plan
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                last_error = exc
                self.logger.warning(
                    "[AutoConfig] Discovery parsing failed on page %s (attempt %s/%s): %s",
                    page_number,
                    attempt,
                    attempts,
                    exc,
                )
                augmented_prompt = self._augment_prompt_for_retry(prompt)

        self.logger.error(
            "[AutoConfig] Discovery exhausted retries for page %s. Last error: %s",
            page_number,
            last_error,
        )
        return None

    def _limit_fields(self, fields: List[FieldDiscoveryEntry]) -> List[FieldDiscoveryEntry]:
        limit = self.per_page_field_limit
        if not limit or len(fields) <= limit:
            return fields
        return fields[: int(limit)]

    def _generate_item_for_candidate(
        self,
        candidate: _FieldCandidate,
        prompt: str,
        image_path: str,
    ) -> Optional[dict]:
        attempts = 2
        augmented_prompt = prompt
        last_error: Optional[Exception] = None

        for attempt in range(1, attempts + 1):
            self.logger.info(
                "[AutoConfig] Materialising field '%s' (attempt %s/%s)",
                candidate.field_name,
                attempt,
                attempts,
            )
            raw_output = self.vlm_infer.infer(image_path, augmented_prompt)
            try:
                parsed = DirtyJsonParser.parse(raw_output)
                if not isinstance(parsed, dict):
                    raise ValueError("Expected a JSON object for ExtractionItem.")

                parsed["field_name"] = candidate.field_name
                parsed["type"] = candidate.type

                # Force document-agnostic output by ignoring explicit page numbers from the model.
                parsed["probable_pages"] = []
                if candidate.has_multipage():
                    parsed["multipage_value"] = True

                item_model = ExtractionItem.model_validate(parsed)
                item_dict = item_model.model_dump()

                item_dict["field_name"] = candidate.field_name
                item_dict["type"] = candidate.type
                item_dict["probable_pages"] = []
                if item_dict.get("scope") == "pages":
                    item_dict["scope"] = "whole"

                if candidate.has_multipage():
                    item_dict["multipage_value"] = True

                if candidate.notes_text:
                    extra = item_dict.get("extra") or {}
                    notes_value = extra.get("auto_config_notes")
                    if isinstance(notes_value, list):
                        notes_list = notes_value + [candidate.notes_text]
                    elif isinstance(notes_value, str) and notes_value:
                        notes_list = [notes_value, candidate.notes_text]
                    else:
                        notes_list = [candidate.notes_text]
                    extra["auto_config_notes"] = notes_list
                    item_dict["extra"] = extra

                return item_dict
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                last_error = exc
                self.logger.warning(
                    "[AutoConfig] ExtractionItem validation failed for '%s' (attempt %s/%s): %s",
                    candidate.field_name,
                    attempt,
                    attempts,
                    exc,
                )
                augmented_prompt = self._augment_prompt_for_retry(prompt)

        self.logger.error(
            "[AutoConfig] Failed to generate ExtractionItem for '%s'. Using fallback. Reason: %s",
            candidate.field_name,
            last_error,
        )
        return self._build_fallback_item(candidate)

    def _augment_prompt_for_retry(self, prompt: str) -> str:
        reminder = "\n\nReminder: respond with JSON that strictly matches the schema."
        if reminder.strip() in prompt:
            return prompt
        return f"{prompt}{reminder}"

    def _build_fallback_item(self, candidate: _FieldCandidate) -> Optional[dict]:
        description = f"Auto-config placeholder for {self._prettify_field_name(candidate.field_name)}."
        item_dict = {
            "field_name": candidate.field_name,
            "description": description,
            "probable_pages": [],
            "type": candidate.type,
            "multipage_value": candidate.has_multipage(),
            "multiline_value": candidate.type in {"bullet-points", "summary", "table"},
            "search_keys": self._default_search_keys(candidate.field_name),
            "parent": [],
            "extra": {},
            "table_header": [],
        }

        if candidate.notes_text:
            item_dict["extra"]["auto_config_notes"] = [candidate.notes_text]

        try:
            item_model = ExtractionItem.model_validate(item_dict)
            return item_model.model_dump()
        except ValidationError as exc:
            self.logger.error(
                "[AutoConfig] Fallback item still failed validation for '%s': %s",
                candidate.field_name,
                exc,
            )
            return None

    def _default_search_keys(self, field_name: str) -> List[str]:
        words: List[str] = []
        current = ""
        for char in field_name:
            if char.isupper() and current:
                words.append(current)
                current = char
            else:
                current += char
        if current:
            words.append(current)
        phrase = " ".join(words)
        result = [field_name]
        if phrase and phrase.lower() != field_name.lower():
            result.append(phrase)
        return result

    def _prettify_field_name(self, field_name: str) -> str:
        parts = self._default_search_keys(field_name)
        return parts[1] if len(parts) > 1 else field_name

    def _format_completed_items(self, items: Iterable[dict]) -> List[Dict[str, object]]:
        formatted: List[Dict[str, object]] = []
        for item in items:
            formatted.append(
                {
                    "field_name": item.get("field_name"),
                    "type": item.get("type"),
                    "probable_pages": item.get("probable_pages", []),
                    "multipage_value": item.get("multipage_value", False),
                    "notes": (item.get("extra") or {}).get("auto_config_notes"),
                }
            )
        return formatted

    def _write_output(self, items: List[dict], output_path: str) -> None:
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with output_path_obj.open("w", encoding="utf-8") as fh:
            json.dump(items, fh, indent=2, ensure_ascii=False)
