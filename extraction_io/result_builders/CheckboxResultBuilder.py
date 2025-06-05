from typing import List, Optional, Dict, Any
from extraction_io.ExtractionOutputs import CheckboxOutput, PointFragment


class CheckboxResultBuilder:
    """
    Encapsulates assembling and validating a CheckboxOutput.

    Accepts a list of page‐level fragments, each with keys:
      - "selected_option" (str or None)
      - "selected_options" (List[str] or None)
      - "continue_next_page" (bool)
      - "page_number" (int)

    Combines them into one final CheckboxOutput.
    """

    @staticmethod
    def build(
            field_name: str,
            fragments: List[Dict[str, Any]],
            key: str,
            multipage: bool
    ) -> CheckboxOutput:
        """
        Given page‐level checkbox fragments and the scope ("single_value" or "multi_value"),
        produce a final CheckboxOutput.

        Args:
          field_name: logical name of the checkbox field.
          fragments:  list of dicts, each containing:
            {
              "selected_option": Optional[str],
              "selected_options": Optional[List[str]],
              "continue_next_page": bool,
              "page_number": int
            }
          scope:      either "single_value" or "multi_value".

        Returns:
          A validated CheckboxOutput whose:
            - For single_value: `selected_option` is the first non-empty option found
              (or "" if none), and `continue_next_page` is taken from the last fragment.
            - For multi_value: `selected_options` is the union of all lists found across pages
              (or [] if none), and `continue_next_page` is taken from the last fragment.
        """
        # Determine the final continue_next_page from the last fragment
        gathered = []
        idx = 0
        for f in fragments:
            if f["selected_option"]:
                gathered.append(PointFragment(value=f["selected_option"], page_number=f["page_number"], index=idx))
                idx += 1
            elif f["selected_options"]:
                for o in f["selected_options"]:
                    gathered.append(PointFragment(value=o, page_number=f["page_number"], index=idx))
                    idx += 1


        return CheckboxOutput(
            field_name=field_name,
            value=gathered,
            key=key
        )
