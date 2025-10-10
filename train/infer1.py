from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import torch
from PIL import Image
from transformers import AutoProcessor

from train.model import QwenExtractionModel
from train.utils import ConfigLoader, create_logger


class SingleInferenceApp:
    """
    Run Qwen2.5-VL inference for a single ExtractionItem sample.
    """

    def __init__(self, config_path: str, checkpoint_path: str):
        self.config_loader = ConfigLoader(config_path)
        cfg = self.config_loader.as_dict()

        self.model_cfg: Dict = cfg.get("model", {})
        self.data_cfg: Dict = cfg.get("data", {})
        self.infer_cfg: Dict = cfg.get("inference", {})

        self.logger = create_logger("infer1", log_dir=self.infer_cfg.get("log_dir"))
        self.device = torch.device(self.infer_cfg.get("device", "mps"))

        processor_name = self.data_cfg.get("processor_name")
        if not processor_name:
            raise ValueError("`processor_name` must be defined in data configuration.")
        self.processor = AutoProcessor.from_pretrained(processor_name)
        self.processor.image_processor.do_thumbnail = False

        self.text_cfg = self.data_cfg.get("text") or {}
        self.max_input_tokens = int(self.text_cfg.get("max_input_tokens", 2048))

        self.model_wrapper = QwenExtractionModel(self.model_cfg)
        checkpoint = torch.load(Path(checkpoint_path), map_location=self.device)
        self.model_wrapper.model.load_state_dict(checkpoint["model_state"])
        self.model_wrapper.eval()

        self.generation_cfg = self.infer_cfg.get("generation", {})
        self.logger.info("Loaded checkpoint from %s", checkpoint_path)

    def run(self, sample_path: str) -> str:
        sample = self._load_sample(sample_path)
        image = Image.open(sample["image_path"]).convert("RGB")
        text_input = self._build_input_text(sample["extraction_item"], sample.get("last_page_value"))

        inputs = self.processor(
            images=image,
            text=text_input,
            truncation=True,
            max_length=self.max_input_tokens,
            return_tensors="pt",
        ).to(self.device)

        generation = self.model_wrapper.generate(inputs, self.generation_cfg)
        output_text = self.processor.batch_decode(generation, skip_special_tokens=True)[0]
        return output_text

    def _load_sample(self, sample_path: str) -> Dict:
        with Path(sample_path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        image_rel = payload.get(self.data_cfg.get("dataset", {}).get("image_key", "page_image"))
        if not image_rel:
            raise KeyError("Sample missing page image reference.")
        image_path = (Path(sample_path).parent / image_rel).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        extraction_item = payload.get(self.data_cfg.get("dataset", {}).get("extraction_item_key", "extraction_item")) or {}
        extraction_output = payload.get(self.data_cfg.get("dataset", {}).get("extraction_output_key", "extraction_output")) or {}
        last_page_value: Optional[Dict] = payload.get(self.data_cfg.get("dataset", {}).get("last_page_value_key", "last_page_value"))

        return {
            "image_path": str(image_path),
            "extraction_item": extraction_item,
            "extraction_output": extraction_output,
            "last_page_value": last_page_value,
        }

    def _build_input_text(self, extraction_item: Dict, last_page_value: Optional[Dict]) -> str:
        payload = {"extraction_item": extraction_item}
        if last_page_value is not None:
            payload["last_page_value"] = last_page_value
        return json.dumps(payload, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-sample inference for Qwen extraction model.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--sample", type=str, required=True)
    args = parser.parse_args()

    app = SingleInferenceApp(args.config, args.checkpoint)
    output = app.run(args.sample)
    print(output)


if __name__ == "__main__":
    main()

