from typing import List, Dict, Any, Optional

from extraction_io.ExtractionOutputs import (
    EntityBlockOutput,
    EntityBlockFragment,
    EntityFieldEntry,
    BoundingBox,
)


class EntityBlockResultBuilder:
    """
    Assemble EntityBlockOutput objects from per-page fragments returned by ParseEntityBlock.
    """

    @staticmethod
    def _normalise_bbox(bbox: Any) -> Optional[BoundingBox]:
        if bbox is None:
            return None
        if isinstance(bbox, BoundingBox):
            return bbox
        if isinstance(bbox, dict):
            # Ensure keys exist; allow ints/floats
            return BoundingBox(**bbox)
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            return BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3])
        # Unsupported structure; ignore to avoid validation errors
        return None

    @staticmethod
    def _normalise_fields(fields: Any) -> List[EntityFieldEntry]:
        normalised: List[EntityFieldEntry] = []
        if not fields:
            return normalised
        if isinstance(fields, EntityFieldEntry):
            return [fields]
        if isinstance(fields, list):
            for item in fields:
                if isinstance(item, EntityFieldEntry):
                    normalised.append(item)
                    continue
                if isinstance(item, dict):
                    key = item.get("key") or item.get("label")
                    if not key:
                        continue
                    value = item.get("value", "")
                    normalised.append(
                        EntityFieldEntry(
                            key=str(key),
                            value=str(value) if value is not None else "",
                            confidence=item.get("confidence"),
                        )
                    )
        return normalised

    @staticmethod
    def build(
        field_name: str,
        fragments: List[Dict[str, Any]],
        key: str,
        multipage: bool,
        *args,
        **kwargs
    ) -> EntityBlockOutput:
        fragments = fragments or []

        normalised_fragments: List[EntityBlockFragment] = []
        for frag in fragments:
            if not isinstance(frag, dict):
                continue
            bbox = EntityBlockResultBuilder._normalise_bbox(frag.get("bbox"))
            fields = EntityBlockResultBuilder._normalise_fields(frag.get("fields"))
            fragment_payload = {
                "page_number": frag.get("page_number"),
                "value": frag.get("value") or "",
                "bbox": bbox,
                "confidence": frag.get("confidence"),
                "continue_next_page": frag.get("continue_next_page"),
                "fields": fields,
            }
            if fragment_payload["page_number"] is None:
                continue
            normalised_fragments.append(EntityBlockFragment(**fragment_payload))

        primary_fragment: Optional[EntityBlockFragment] = None
        for frag in normalised_fragments:
            if frag.value:
                primary_fragment = frag
                break
        if primary_fragment is None and normalised_fragments:
            primary_fragment = normalised_fragments[0]

        value = primary_fragment.value if primary_fragment else ""
        page_number = primary_fragment.page_number if primary_fragment else None
        bbox = primary_fragment.bbox if primary_fragment else None
        confidence = primary_fragment.confidence if primary_fragment else None

        aggregated_fields: Dict[str, EntityFieldEntry] = {}
        for frag in normalised_fragments:
            for entry in frag.fields or []:
                if entry.key not in aggregated_fields or not aggregated_fields[entry.key].value:
                    aggregated_fields[entry.key] = entry

        fields = list(aggregated_fields.values())

        fragments_payload = normalised_fragments if (multipage and normalised_fragments) else None

        return EntityBlockOutput(
            field_name=field_name,
            value=value,
            key=key,
            page_number=page_number,
            bbox=bbox,
            confidence=confidence,
            fields=fields,
            fragments=fragments_payload,
        )
