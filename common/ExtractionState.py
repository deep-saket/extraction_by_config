# common/extraction_state.py
from dataclasses import dataclass, field
from typing import List, Tuple, Any, Union
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

    @classmethod
    def reset(cls):
        """
        Clear images, embeddings, and entries for a new extraction cycle.
        """
        cls.images = []
        cls.embeddings = []
        cls.response = []
        cls.extraction_config = None

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
    def get_extraction_items(cls):
        return cls.extraction_items

    @classmethod
    def get_extraction_item(cls, idx: int):
        return cls.extraction_items[idx]

    @classmethod
    def get_current_extraction_item(cls):
        return cls.current_extraction_item
