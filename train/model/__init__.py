"""
Model architectures for schema generation.
"""

from .qwen_extraction_model import QwenExtractionModel
from .qwen3_vl_model import Qwen3VLModel

__all__ = [
    "QwenExtractionModel",
    "Qwen3VLModel",
]
