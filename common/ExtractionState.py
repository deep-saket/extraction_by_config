# common/extraction_state.py
from dataclasses import dataclass, field
from typing import List, Tuple, Any
import torch
from extraction_io.ExtractionOutputs import ExtractionOutput


@dataclass
class ExtractionState:
    """
    Holds global images, embeddings, and extraction entries state for a document extraction cycle.
    Uses class variables to maintain state.
    """
    images: List[Tuple[int, str]] = field(default_factory=list)
    embeddings: List[Tuple[int, torch.Tensor]] = field(default_factory=list)
    entries: List[ExtractionOutput] = field(default_factory=list)  # Holds raw extraction entries or validated models

    @classmethod
    def reset(cls):
        """
        Clear images, embeddings, and entries for a new extraction cycle.
        """
        cls.images = []
        cls.embeddings = []
        cls.entries = []

    @classmethod
    def set_images(cls, imgs):
        cls.images = imgs

    @classmethod
    def set_embeddings(cls, embs):
        cls.embeddings = embs

    @classmethod
    def add_entry(cls, entry: Any):
        """
        Add a single extraction entry (could be a dict or Pydantic model).
        """
        cls.entries.append(entry)

    @classmethod
    def set_entries(cls, entries_list: List[Any]):
        """
        Replace the entire entries list with a new list.
        """
        cls.entries = entries_list

    @classmethod
    def get_images(cls):
        return cls.images

    @classmethod
    def get_embeddings(cls):
        return cls.embeddings

    @classmethod
    def get_entries(cls):
        return cls.entries
