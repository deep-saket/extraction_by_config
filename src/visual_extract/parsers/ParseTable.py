from src.visual_extract.parsers.ParseBase import ParseBase
from typing import List, Any, Dict, Set, Tuple
from extraction_io.generation_utils import TableGeneration
from common import ExtractionState


class ParseTable(ParseBase):
    """
    Uses a Vision-Language Model (VLM) to extract tables from document images/pages.
    Inherit from ParserBase for consistency with other parser types.
    """

    def _choose_schema(self) -> Dict[str, Any]:
        # Return the JSON schema dict for this extraction type
        return self.parser_response_model.model_json_schema()

    def _process_page(self, page_num: int, prev_value) -> Any:
        image_path = ExtractionState.get_image(page_num)
        if image_path is None:
            return None
        schema_dict = self._choose_schema()
        prompt = self.prompt_builder.build(
            self.item,
            schema_dict=schema_dict,
            prev_value=prev_value,
        )
        gen = self.vlm_processor(image_path, prompt, TableGeneration)
        # Attach page number info to returned model for aggregation
        return {'gen': gen, 'page': page_num}

    def _get_prev_page_context(self, prev_page_res):
        if prev_page_res and getattr(prev_page_res, 'rows', None):
            first_row = prev_page_res.rows[0]
            if isinstance(first_row, list):
                return " | ".join(first_row)
            return str(first_row)
        return None

    def _row_signature(self, row_values: List[str]) -> Tuple:
        """Build a content-based signature for a row to avoid duplicates across pages."""
        return tuple((value or "").strip() for value in row_values)

    def run(self, pages: List[int]) -> Any:
        # We'll return a list of fragments, each: { 'page_number': int, 'rows': [[...values...]], 'columns': [...]? }
        prev_value = None
        consumed_pages: Set[int] = set()

        for start_pg in pages or []:
            if start_pg in consumed_pages:
                continue

            fragments_for_table: List[Dict[str, Any]] = []
            seen_rows: Set[Tuple] = set()
            cur_pg = start_pg
            first_fragment = True

            while True:
                page_result = self._process_page(cur_pg, prev_value)
                if not page_result or not page_result.get('gen'):
                    break
                gen = page_result['gen']

                # Build this page's rows as list of values, de-duped by signature
                page_rows_values: List[List[str]] = []
                for row_values in getattr(gen, 'rows', []) or []:
                    values = [str(v) if v is not None else '' for v in row_values]
                    sig = self._row_signature(values)
                    if sig in seen_rows:
                        continue
                    seen_rows.add(sig)
                    page_rows_values.append(values)

                if page_rows_values:
                    frag: Dict[str, Any] = {
                        'page_number': cur_pg,
                        'rows': page_rows_values,
                    }
                    columns_from_gen = getattr(gen, 'columns', None) or []
                    if columns_from_gen:
                        frag['columns'] = [str(col) for col in columns_from_gen]
                    elif first_fragment and getattr(self.item, 'table_header', None):
                        if any(str(h).strip() for h in self.item.table_header):
                            frag['columns'] = list(self.item.table_header)
                    fragments_for_table.append(frag)

                prev_value = self._get_prev_page_context(gen)
                consumed_pages.add(cur_pg)

                # Continue if VLM indicates table spans next page
                if getattr(gen, 'continue_next_page', False):
                    cur_pg += 1
                    first_fragment = False
                    continue
                else:
                    break

            # If we found any fragments for this starting page, that's our table; stop scanning further pages
            if fragments_for_table:
                return fragments_for_table

        # If nothing found at all, return empty list
        return []
