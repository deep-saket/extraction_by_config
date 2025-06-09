from typing import List, Optional, Dict, Any
from extraction_io.ExtractionOutputs import SummaryOutput


class SummaryResultBuilder:
    """
    Encapsulates assembling and validating a SummaryOutput.
    """

    @staticmethod
    def build(
        field_name: str,
        fragments: str,
        *args,
        page_range: Optional[List[int]] = None,
        related_fields: Optional[List[str]] = None,
        **kwargs
    ) -> SummaryOutput:
        """
        Given the concatenated summary text and optional metadata,
        produce a validated SummaryOutput.

        Args:
          field_name:     logical name or identifier for this summary.
          summary:        the final summary text (all fragments concatenated).
          page_range:     [start_page, end_page] or None.
          related_fields: list of extracted field names summarized, or None.
        """
        summary = fragments.summary['summary']
        return SummaryOutput(
            field_name=field_name,
            value=summary,
            page_range=page_range,
            related_fields=related_fields,
            key=""
        )