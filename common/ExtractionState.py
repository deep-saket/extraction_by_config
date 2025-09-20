# common/extraction_state.py
from dataclasses import dataclass, field
from typing import List, Tuple, Any, Union, Dict
import torch
from extraction_io.ExtractionOutputs import ExtractionOutput
from extraction_io.ExtractionItems import ExtractionItems
from common.BaseComponent import BaseComponent


@dataclass
class ExtractionState(BaseComponent):
    """
    Holds global images, embeddings, and extraction entries state for a document extraction cycle.
    Uses class variables to maintain state.
    """
    extraction_items: Union[List[dict], ExtractionItems]
    current_extraction_item: ExtractionItems
    images: List[Tuple[int, str]] = field(default_factory=list)
    embeddings: List[Tuple[int, torch.Tensor]] = field(default_factory=list)
    response: List[ExtractionOutput] = field(default_factory=list)  # Holds raw extraction entries or validated models
    checkboxes: Dict[int, dict] = field(default_factory=dict)

    
    @classmethod
    def reset(cls):
        """
        Clear images, embeddings, and entries for a new extraction cycle.
        """
        cls.images = []
        cls.embeddings = []
        cls.response = []
        cls.extraction_config = None
        cls.checkboxes = {}

    @classmethod
    def update_curr_extraction_item(cls, idx: int):
        cls.current_extraction_item = cls.extraction_items[idx]

    @classmethod
    def set_images(cls, imgs):
        cls.images = imgs

    @classmethod
    def set_embeddings(cls, embs):
        cls.embeddings = embs

    @classmethod
    def add_response(cls, entry: Any):
        """
        Add a single extraction entry (could be a dict or Pydantic model).
        """
        cls.response.append(entry)

    @classmethod
    def set_responses(cls, response_list: List[Any]):
        """
        Replace the entire entries list with a new list.
        """
        cls.response = response_list

    @classmethod
    def set_extraction_items(cls, extraction_items: Union[List[dict], ExtractionItems]):
        cls.extraction_items = extraction_items

    @classmethod
    def get_images(cls):
        return cls.images

    @classmethod
    def get_embeddings(cls):
        return cls.embeddings

    @classmethod
    def get_responses(cls):
        return cls.response

    
    @classmethod
    def get_response_by_field_name(cls, field_name: str) -> Any:
        """
        Get response item matching the given field name.

        Args:
            field_name: The field name to search for

        Returns:
            Matching response item or None if not found
        """
        matches = [r for r in cls.response if hasattr(r, 'root') and hasattr(r.root, 'field_name')  and r.root.field_name == field_name]
        return matches[0] if matches else None

    @classmethod
    def get_response_by_field(cls, field_name: str):
        """
        Retrieve the output (ExtractionOutput) for a given field_name from response list.
        Returns None if not found.
        """
        for entry in cls.response:
            if hasattr(entry.root, 'field_name') and entry.root.field_name == field_name:
                return entry.root
        return None

    @classmethod
    def get_extraction_items(cls):
        return cls.extraction_items

    @classmethod
    def get_extraction_item(cls, idx: int):
        return cls.extraction_items[idx]

    @classmethod
    def get_extraction_item_by_fieldname(cls, field_name: str) -> Any:
        """
        Get extraction item matching the given field name.

        Args:
            field_name: The field name to search for

        Returns:
            Matching extraction item or None if not found
        """
        matches = [item for item in cls.extraction_items if
                   hasattr(item, 'field_name') and item.field_name == field_name]
        return matches[0] if matches else None

    @classmethod
    def get_current_extraction_item(cls):
        return cls.current_extraction_item

    @classmethod
    def set_checkboxes(cls, boxes: Dict[int, dict]):
        cls.checkboxes = boxes

    @classmethod
    def get_checkboxes(cls):
        return cls.checkboxes

    @classmethod
    def has_checkboxes(cls) -> bool:
        """
        Check if there are any checkboxes defined.
        Returns:
            bool: True if there are checkboxes, False otherwise
        """
        return len(cls.checkboxes) > 0
