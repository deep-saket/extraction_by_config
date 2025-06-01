from typing import List, Optional, Dict, Any
from extraction_io.ExtractionOutputs import (
    KeyValueOutput,
    KVFragment,
    BulletPointsOutput,
    BulletPoint
)


class BulletPointsResultBuilder:
    """
    Encapsulates assembling and validating a BulletPointsOutput.
    """

    @staticmethod
    def build(
        field_name: str,
        fragments: List[Dict[str, Any]],
        key: str,
        multipage: bool
    ) -> BulletPointsOutput:
        """
        Given a flat list of bullet dicts (each with "value",
        "post_processing_value", "page_number", "point_number"),
        produce a validated BulletPointsOutput.

        Args:
          field_name: logical name of the field.
          fragments:    list of dicts, each matching BulletPoint's schema.
          key:        the literal search key (often same as description).
        """
        # 1) Build a list of BulletPoint instances
        bp_models = [BulletPoint(**b) for b in fragments]

        # 2) Construct and return a validated BulletPointsOutput model
        return BulletPointsOutput(
            field_name=field_name,
            points=bp_models,
            key=key
        )