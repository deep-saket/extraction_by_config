"""
Dataset and dataloader utilities for schema generation training.
"""

from .single_page_dataset import ExtractionSampleDataset, ExtractionCollator

__all__ = [
    "ExtractionSampleDataset",
    "ExtractionCollator",
]
