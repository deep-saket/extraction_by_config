from dataclasses import dataclass, field
from typing import List, Tuple, Any, Dict
import torch
from common import BaseComponent


@dataclass
class ClassificationState(BaseComponent):
    """
    Holds images, embeddings, and classification context for a routing cycle.
    Uses class variables to share state across helpers.
    """
    classification_config: Any = None
    images: List[Tuple[int, str]] = field(default_factory=list)
    embeddings: List[Tuple[int, torch.Tensor]] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def reset(cls):
        """
        Clear images, embeddings, and scores for a new classification cycle.
        """
        cls.images = []
        cls.embeddings = []
        cls.classification_config = None
        cls.scores = {}

    @classmethod
    def set_images(cls, imgs: List[Tuple[int, str]]):
        cls.images = imgs

    @classmethod
    def set_embeddings(cls, embs: List[Tuple[int, torch.Tensor]]):
        cls.embeddings = embs

    @classmethod
    def set_config(cls, cfg: Any):
        cls.classification_config = cfg

    @classmethod
    def set_score(cls, candidate_file: str, score: float):
        cls.scores[candidate_file] = score

    @classmethod
    def get_images(cls):
        return cls.images

    @classmethod
    def get_embeddings(cls):
        return cls.embeddings

    @classmethod
    def get_scores(cls):
        return cls.scores

    @classmethod
    def get_config(cls):
        return cls.classification_config
