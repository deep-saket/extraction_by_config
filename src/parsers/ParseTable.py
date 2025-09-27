from src.parsers.ParseBase import ParseBase
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
            # Provide the first row as context cue for the next-page prompt, if any
            return prev_page_res.rows[0]
        return None

    def _row_signature(self, row) -> Tuple:
        """Build a content-based signature for a row to avoid duplicates across pages."""
        try:
            cols = sorted((c.col_number, (c.value or '').strip()) for c in (row.cols or []))
            return tuple(cols)
        except Exception:
            return (str(row),)

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
                for row in getattr(gen, 'rows', []) or []:
                    sig = self._row_signature(row)
                    if sig in seen_rows:
                        continue
                    seen_rows.add(sig)
                    values = [str(c.value) if c.value is not None else '' for c in sorted(row.cols, key=lambda x: x.col_number)]
                    page_rows_values.append(values)

                if page_rows_values:
                    frag: Dict[str, Any] = {
                        'page_number': cur_pg,
                        'rows': page_rows_values,
                    }
                    # Include columns only once (first non-empty wins in builder); use config-provided table_header if any
                    if first_fragment and getattr(self.item, 'table_header', None):
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
