from src.parsers.ParseBase import ParseBase
from extraction_io.ExtractionItems import ExtractionItem
from extraction_io.generation_utils.SummaryGeneration import SummaryGeneration
from common import ExtractionState
from typing import List, Any
from src.helper.ParentProcessor import ParentProcessor

class ParseSummary(ParseBase):
    """
    Parser for summary extraction supporting all scopes: whole, section, pages, extraction_items.
    It uses the provided VLMProcessor for page/image summarization and delegates
    parent-based summarization to a configured parent processor via ParentProcessor.
    """
    def __init__(self, item: ExtractionItem, vlm_processor, prompt_builder, parser_response_model):
        super().__init__(item, vlm_processor, prompt_builder, parser_response_model)
        self.parent_processor = ParentProcessor()

    def _choose_schema(self):
        return self.parser_response_model_schema

    def _process_page(self, page_num: int, prev_value: str) -> Any:
        # For summaries we return a SummaryGeneration model for the page
        image_path = self.vlm_processor.pdf_processor.get_page_image(page_num)
        if image_path is None:
            return None
        schema_dict = SummaryGeneration.model_json_schema()
        prompt = self.prompt_builder.build(self.item, schema_dict=schema_dict, prev_value=prev_value)
        gen = self.vlm_processor(image_path, prompt, SummaryGeneration)
        return gen

    def run(self, pages: List[int]) -> SummaryGeneration:
        item = self.item
        scope = item.scope

        if scope == "extraction_items":
            # Gather parent outputs from ExtractionState and concatenate
            parent_texts = []
            for parent_name in item.parent:
                resp = ExtractionState.get_response_by_field_name(parent_name)
                if resp is not None:
                    # Convert parent response into text
                    text = self._parent_to_text(resp.root if hasattr(resp, 'root') else resp)
                    parent_texts.append(text)
            raw_concat = "\n\n".join(parent_texts)

            # Delegate to ParentProcessor which will pick configured processor (e.g., ExtractionItemsSummariser)
            result = self.parent_processor(raw_concat)
            # ParentProcessor returns either transformed raw_data or a Pydantic model depending on processor
            if isinstance(result, SummaryGeneration):
                return result
            # If processor returned a raw string or dict, wrap into SummaryGeneration
            if isinstance(result, str):
                return SummaryGeneration(field_name=item.field_name, summary=result, continue_next_page=False)
            try:
                # attempt to coerce dict
                return SummaryGeneration.model_validate(result)
            except Exception:
                return SummaryGeneration(field_name=item.field_name, summary=str(result), continue_next_page=False)

        # For page-based scopes (whole/section/pages)
        fragments: List[str] = []
        prev_value = ""
        for p in pages:
            gen = self._process_page(p, prev_value)
            if gen is None:
                continue
            frag_text = gen.summary if isinstance(gen.summary, str) else (gen.summary.get('summary') if isinstance(gen.summary, dict) else str(gen.summary))
            fragments.append(frag_text)
            prev_value = "\n".join(fragments)
            if not getattr(gen, 'continue_next_page', False):
                break

        final_summary = "\n\n".join(fragments).strip()
        return SummaryGeneration(field_name=item.field_name, summary=final_summary, continue_next_page=False)

    def _parent_to_text(self, parent_obj: Any) -> str:
        if parent_obj is None:
            return ""
        obj = parent_obj
        # If it's a RootModel wrapper, try to access .root
        if hasattr(obj, 'root'):
            obj = obj.root
        if hasattr(obj, 'value') and isinstance(obj.value, str):
            return obj.value
        if hasattr(obj, 'value') and isinstance(obj.value, list):
            parts = []
            for pt in obj.value:
                try:
                    parts.append(pt.value)
                except Exception:
                    parts.append(str(pt))
            return "\n".join(parts)
        if hasattr(obj, 'rows'):
            rows = []
            for r in obj.rows:
                try:
                    rows.append(str(r.row))
                except Exception:
                    rows.append(str(r))
            return "\n".join(rows)
        return str(obj)
