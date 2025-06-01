from typing import List, Optional, Dict, Any
from extraction_io.ExtractionOutputs import (
    KeyValueOutput,
    KVFragment,
    BulletPointsOutput,
    BulletPoint
)

class KeyValueResultBuilder:
    """
    Encapsulates assembling and validating a KeyValueOutput.
    """

    @staticmethod
    def build(
        field_name: str,
        fragments: List[Dict[str, Any]],
        key: str,
        multipage: bool
    ) -> KeyValueOutput:
        """
        Given a list of page‐level fragments (each with keys "value",
        "post_processing_value", "page_number"), produce a KeyValueOutput.

        Args:
          field_name: logical name of the field.
          fragments: list of dicts, each matching KVFragment's schema.
          key:       the literal search key (often same as description).
          multipage: whether this field spanned multiple pages.
        """
        # 1) Concatenate all fragment values into one final string
        concatenated_value = "".join(f["value"] for f in fragments)

        # 2) If any fragment had post_processing_value, concatenate those too
        concatenated_post = None
        if any(f.get("post_processing_value") for f in fragments):
            concatenated_post = "".join(
                f.get("post_processing_value", "") for f in fragments
            )

        # 3) The “page_number” in the top‐level KeyValueOutput is the first page
        first_page = fragments[0]["page_number"] if fragments else None

        # 4) Build the list of KVFragment instances (if multipage)
        kvfrags = None
        if multipage and fragments:
            kvfrags = [KVFragment(**f) for f in fragments]

        # 5) Construct and return a validated KeyValueOutput model
        return KeyValueOutput(
            field_name=field_name,
            value=concatenated_value,
            post_processing_value=concatenated_post,
            page_number=first_page,
            key=key,
            multipage_detail=kvfrags,
        )
