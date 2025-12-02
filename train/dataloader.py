from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from torch.utils.data import DataLoader, random_split

from transformers import AutoProcessor

from train.mock_components import MockProcessor

from train.data_loader import ExtractionSampleDataset, ExtractionCollator


@dataclass
class LoaderBundle:
    train: DataLoader
    val: Optional[DataLoader]
    test: Optional[DataLoader]


class DataModule:
    """
    Prepare dataset and dataloaders for both supervised and RL objectives.
    """

    def __init__(self, data_cfg: Dict):
        self.data_cfg = dict(data_cfg)
        self.batch_size = int(self.data_cfg.get("batch_size", 1))
        self.num_workers = int(self.data_cfg.get("num_workers", 0))
        self.shuffle = bool(self.data_cfg.get("shuffle", True))
        self.train_split = float(self.data_cfg.get("train_split", 0.8))
        self.val_split = float(self.data_cfg.get("val_split", 0.1))
        self.test_split = float(self.data_cfg.get("test_split", 0.1))

        processor_name = self.data_cfg.get("processor_name")
        if not processor_name:
            raise ValueError("`processor_name` must be specified in data configuration.")

        if processor_name == "mock":
            self.processor = MockProcessor()
        else:
            self.processor = AutoProcessor.from_pretrained(processor_name)
            if hasattr(self.processor, "image_processor") and hasattr(self.processor.image_processor, "do_thumbnail"):
                self.processor.image_processor.do_thumbnail = False

        text_cfg = self.data_cfg.get("text") or {}
        self.collator = ExtractionCollator(self.processor, text_cfg)

        dataset_cfg = self.data_cfg.get("dataset") or {}
        self.dataset = ExtractionSampleDataset(dataset_cfg)

        total = len(self.dataset)
        val_count = int(total * self.val_split)
        test_count = int(total * self.test_split)
        train_count = total - val_count - test_count
        if train_count <= 0:
            raise ValueError("Invalid split configuration; training split must be > 0.")

        self.train_set, self.val_set, self.test_set = random_split(
            self.dataset,
            lengths=[train_count, val_count, test_count],
            generator=None,
        )

    def construct_loaders(self) -> LoaderBundle:
        train_loader = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collator,
        )

        val_loader = DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collator,
        ) if len(self.val_set) > 0 else None

        test_loader = DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collator,
        ) if len(self.test_set) > 0 else None

        return LoaderBundle(train=train_loader, val=val_loader, test=test_loader)
