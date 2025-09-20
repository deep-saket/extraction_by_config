from typing import List, Optional, Any
from extraction_io.ExtractionOutputs import SummaryOutput
from extraction_io.generation_utils.SummaryGeneration import SummaryGeneration


class SummaryResultBuilder:
    """
    Encapsulates assembling and validating a SummaryOutput.
    """

    @staticmethod
    def _extract_text_from_generation(gen: Any) -> str:
        if gen is None:
            return ""
        if isinstance(gen, SummaryGeneration):
            summary_field = gen.summary
        elif isinstance(gen, dict):
            summary_field = gen.get('summary', '')
        else:
            # could be a string
            return str(gen)

        if isinstance(summary_field, str):
            return summary_field
        if isinstance(summary_field, dict):
            return summary_field.get('summary', '')
        return str(summary_field)

    @staticmethod
    def build(
        field_name: str,
        fragments: Any,
        *args,
        page_range: Optional[List[int]] = None,
        related_fields: Optional[List[str]] = None,
        key: str = "",
        **kwargs
    ) -> SummaryOutput:
        """
        Given the SummaryGeneration (or raw string) and optional metadata,
        produce a validated SummaryOutput.
        """
        summary_text = SummaryResultBuilder._extract_text_from_generation(fragments)
        return SummaryOutput(
            field_name=field_name,
            value=summary_text,
            key=key,
            page_range=page_range,
            related_fields=related_fields
        )