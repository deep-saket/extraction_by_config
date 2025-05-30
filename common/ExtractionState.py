# common/extraction_state.py
from dataclasses import dataclass, field
from typing import List, Tuple
import torch


@dataclass
class ExtractionState:
    """
    Holds global images and embeddings state for a document extraction cycle.
    Uses class variables to maintain state.
    """
    images: List[Tuple[int, str]] = field(default_factory=list)
    embeddings: List[Tuple[int, torch.Tensor]] = field(default_factory=list)

    @classmethod
    def reset(cls):
        """
        Clear images and embeddings for a new extraction cycle.
        """
        cls.images = []
        cls.embeddings = []

    @classmethod
    def set_images(cls, imgs):
        cls.images = imgs

    @classmethod
    def set_embeddings(cls, embs):
        cls.embeddings = embs

    @classmethod
    def get_images(cls):
        return cls.images

    @classmethod
    def get_embeddings(cls):
        return cls.embeddings