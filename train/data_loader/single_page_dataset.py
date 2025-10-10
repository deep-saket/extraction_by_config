from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image
from torch.utils.data import Dataset


class ExtractionSampleDataset(Dataset):
    """
    Dataset that exposes single-page samples consisting of:
      - page_image:       path to the document page image.
      - extraction_item:  JSON describing the ExtractionItem to process.
      - extraction_output: Expected ExtractionOutput JSON.
      - last_page_value:  Optional JSON payload with context from previous pages.
    """

    def __init__(self, data_cfg: Dict):
        self.data_cfg = dict(data_cfg)
        self.root_dir = Path(self.data_cfg.get("root_dir", ".")).expanduser().resolve()
        if not self.root_dir.is_dir():
            raise NotADirectoryError(f"Dataset directory not found: {self.root_dir}")

        self.image_key = self.data_cfg.get("image_key", "page_image")
        self.item_key = self.data_cfg.get("extraction_item_key", "extraction_item")
        self.output_key = self.data_cfg.get("extraction_output_key", "extraction_output")
        self.last_value_key = self.data_cfg.get("last_page_value_key", "last_page_value")
        self.sample_ext = self.data_cfg.get("sample_ext", ".json")

        self._samples: List[Path] = sorted(self.root_dir.glob(f"*{self.sample_ext}"))
        if not self._samples:
            raise ValueError(f"No samples found in {self.root_dir} with extension {self.sample_ext}")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict:
        sample_path = self._samples[idx]
        with sample_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        image_rel_path = payload.get(self.image_key)
        if not image_rel_path:
            raise KeyError(f"Sample missing '{self.image_key}' field: {sample_path}")
        image_path = (sample_path.parent / image_rel_path).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        extraction_item = payload.get(self.item_key) or {}
        extraction_output = payload.get(self.output_key) or {}
        last_page_value: Optional[Dict] = payload.get(self.last_value_key)

        return {
            "image_path": str(image_path),
            "extraction_item": extraction_item,
            "extraction_output": extraction_output,
            "last_page_value": last_page_value,
        }


class ExtractionCollator:
    """
    Collate samples into batched tensors using the provided Hugging Face processor.
    """

    def __init__(self, processor, text_cfg: Dict):
        self.processor = processor
        self.text_cfg = dict(text_cfg)
        self.max_input_tokens = int(self.text_cfg.get("max_input_tokens", 2048))
        self.max_target_tokens = int(self.text_cfg.get("max_target_tokens", 1024))

    def __call__(self, batch: List[Dict]):
        images = [Image.open(sample["image_path"]).convert("RGB") for sample in batch]
        text_inputs = [
            self._build_input_text(sample["extraction_item"], sample.get("last_page_value"))
            for sample in batch
        ]
        text_targets = [
            self._build_output_text(sample["extraction_output"])
            for sample in batch
        ]

        model_inputs = self.processor(
            images=images,
            text=text_inputs,
            truncation=True,
            max_length=self.max_input_tokens,
            return_tensors="pt",
        )

        with self.processor.as_target_processor():
            labels = self.processor(
                text=text_targets,
                truncation=True,
                max_length=self.max_target_tokens,
                return_tensors="pt",
            )["input_ids"]

        model_inputs["labels"] = labels
        return model_inputs

    def _build_input_text(self, extraction_item: Dict, last_page_value: Optional[Dict]) -> str:
        payload = {"extraction_item": extraction_item}
        if last_page_value is not None:
            payload["last_page_value"] = last_page_value
        return json.dumps(payload, ensure_ascii=False)

    def _build_output_text(self, extraction_output: Dict) -> str:
        return json.dumps(extraction_output, ensure_ascii=False)

