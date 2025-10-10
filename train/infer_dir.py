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


class DirectoryInferenceApp:
    """
    Batch inference across a directory of ExtractionItem samples.
    """

    def __init__(self, config_path: str, checkpoint_path: str):
        self.config_loader = ConfigLoader(config_path)
        cfg = self.config_loader.as_dict()

        self.model_cfg: Dict = cfg.get("model", {})
        self.data_cfg: Dict = cfg.get("data", {})
        self.infer_cfg: Dict = cfg.get("inference", {})

        self.logger = create_logger("infer_dir", log_dir=self.infer_cfg.get("log_dir"))
        self.device = torch.device(self.infer_cfg.get("device", "mps"))
        self.output_suffix = self.infer_cfg.get("output_suffix", "_pred")

        processor_name = self.data_cfg.get("processor_name")
        if not processor_name:
            raise ValueError("`processor_name` must be defined in data configuration.")
        self.processor = AutoProcessor.from_pretrained(processor_name)
        self.processor.image_processor.do_thumbnail = False

        text_cfg = self.data_cfg.get("text") or {}
        self.max_input_tokens = int(text_cfg.get("max_input_tokens", 2048))

        self.model_wrapper = QwenExtractionModel(self.model_cfg)
        checkpoint = torch.load(Path(checkpoint_path), map_location=self.device)
        self.model_wrapper.model.load_state_dict(checkpoint["model_state"])
        self.model_wrapper.eval()
        self.generation_cfg = self.infer_cfg.get("generation", {})

        dataset_cfg = self.data_cfg.get("dataset", {})
        self.image_key = dataset_cfg.get("image_key", "page_image")
        self.item_key = dataset_cfg.get("extraction_item_key", "extraction_item")
        self.output_key = dataset_cfg.get("extraction_output_key", "extraction_output")
        self.last_key = dataset_cfg.get("last_page_value_key", "last_page_value")

        self.logger.info("Loaded checkpoint from %s", checkpoint_path)

    def run(self, input_dir: str, output_dir: str) -> None:
        input_path = Path(input_dir).expanduser().resolve()
        output_path = Path(output_dir).expanduser().resolve()
        output_path.mkdir(parents=True, exist_ok=True)

        for sample_file in sorted(input_path.glob("*.json")):
            prediction = self._process_sample(sample_file)
            target_path = output_path / f"{sample_file.stem}{self.output_suffix}.json"
            with target_path.open("w", encoding="utf-8") as handle:
                json.dump(prediction, handle, indent=2, ensure_ascii=False)
            self.logger.info("Saved %s", target_path)

    def _process_sample(self, sample_path: Path) -> Dict:
        with sample_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        image_rel = payload.get(self.image_key)
        if not image_rel:
            raise KeyError(f"Sample missing '{self.image_key}': {sample_path}")
        image = Image.open((sample_path.parent / image_rel).resolve()).convert("RGB")

        extraction_item = payload.get(self.item_key) or {}
        last_value: Optional[Dict] = payload.get(self.last_key)

        text_input = self._build_input_text(extraction_item, last_value)
        inputs = self.processor(
            images=image,
            text=text_input,
            truncation=True,
            max_length=self.max_input_tokens,
            return_tensors="pt",
        ).to(self.device)

        generation = self.model_wrapper.generate(inputs, self.generation_cfg)
        output_text = self.processor.batch_decode(generation, skip_special_tokens=True)[0]
        return {"prediction": output_text}

    def _build_input_text(self, extraction_item: Dict, last_page_value: Optional[Dict]) -> str:
        payload = {"extraction_item": extraction_item}
        if last_page_value is not None:
            payload["last_page_value"] = last_page_value
        return json.dumps(payload, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Directory inference for Qwen extraction model.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    app = DirectoryInferenceApp(args.config, args.checkpoint)
    app.run(args.input, args.output)


if __name__ == "__main__":
    main()

