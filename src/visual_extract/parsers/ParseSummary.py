from src.visual_extract.parsers.ParseBase import ParseBase
from extraction_io.ExtractionItems import ExtractionItem
from common import ExtractionState
from typing import List, Any
import copy


class ParseSummary(ParseBase):
    """
    Parser for summary extraction supporting all scopes: whole, section, pages, extraction_items.
    It uses the provided VLMProcessor for page/image summarization and delegates
    parent-based summarization to a configured parent processor via ParentProcessor.
    """
    def __init__(self, item: ExtractionItem, vlm_processor, prompt_builder, parser_response_model):
        super().__init__(item, vlm_processor, prompt_builder, parser_response_model)

    def _choose_schema(self):
        return self.parser_response_model_schema

    def _process_page(self, page_num: int, prev_value=None) -> Any:
        # For summarising we return a self.parser_response_model model for the page
        image_path = ExtractionState.get_image(page_num)
        if image_path is None:
            return None
        #TODO prompt upliftment required
        item_copy = copy.deepcopy(self.item)
        item_copy.field_name = f"summary_{page_num}"
        item_copy.probable_pages = [page_num]
        prompt = self.prompt_builder.build(item_copy, schema_dict=self._choose_schema())
        summ = self.vlm_processor(image_path, prompt, self.parser_response_model, item=item_copy)
        self.result_builder_factory(item_copy, summ, interim=True)
        return summ

    def _process_pages(self, page_nums: List[int]) -> dict:
        results = {}
        for p in page_nums:
            gen = self._process_page(p)
            if gen is not None:
                results[p] = gen
        return results

    def run(self, pages: List[int]):
        item = self.item
        scope = item.scope
        result = None

        if scope == "extraction_items":
            # Gather parent outputs from ExtractionState and concatenate
            result = {}
            for parent_name in item.parent:
                resp = ExtractionState.get_response_by_field_name(parent_name)
                if resp is not None:
                    # Convert parent response into text
                    text = self._parent_to_text(resp.root if hasattr(resp, 'root') else resp)
                    result[resp.root.field_name] = text

        if scope == "whole":
            pages = [img[0] for img in ExtractionState.get_images()]
            summaries_detail = self._process_pages(pages)
            # Pass summaries to parent processor
            result = {}
            for pg_num, summ in summaries_detail.items():
                # Pass summaries to parent processor
                self.item.parent.append(summ.field_name)
                result[pg_num] = summ.value

        # Handle scope == pages
        if scope == "pages":
            summaries_detail = self._process_pages(pages)
            # Pass summaries to parent processor
            result = {}
            for pg_num, summ in summaries_detail.items():
                # Pass summaries to parent processor
                self.item.parent.append(summ.field_name)
                result[pg_num] = summ.value

        if scope == "section":
            summaries_detail = self._process_pages(pages)
            # Pass summaries to parent processor
            result = {}
            for pg_num, summ in summaries_detail.items():
                # Pass summaries to parent processor
                self.item.parent.append(summ.field_name)
                result[pg_num] = summ.value

        # Default behavior: process pages normally
        # ParentProcessor returns either transformed raw_data or a Pydantic model depending on processor
        return result

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
