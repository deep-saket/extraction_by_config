from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from train.mock_components import MockVisionLanguageModel


class QwenExtractionModel:
    """
    Thin wrapper that loads a Qwen2.5-VL model for direct ExtractionItem → ExtractionOutput generation.
    """

    def __init__(self, model_cfg: Dict):
        self.model_cfg = dict(model_cfg)
        self.model_name = self.model_cfg.get("pretrained_name")
        if not self.model_name:
            raise ValueError("`pretrained_name` must be provided in model configuration.")

        self.dtype = getattr(torch, self.model_cfg.get("dtype", "float16"))
        self.device = torch.device(self.model_cfg.get("device", "mps"))
        self.gradient_checkpointing = bool(self.model_cfg.get("gradient_checkpointing", True))

        if self.model_name == "mock" or self.model_cfg.get("mock", False):
            mock_vocab_size = int(self.model_cfg.get("mock_vocab_size", 512))
            mock_hidden = int(self.model_cfg.get("mock_hidden_size", 64))
            mock_image = int(self.model_cfg.get("mock_image_size", 32))
            self.model = MockVisionLanguageModel(
                vocab_size=mock_vocab_size,
                hidden_size=mock_hidden,
                image_size=mock_image,
            )
            self.tokenizer = None
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                device_map="auto" if self.device.type != "cpu" else None,
            )
            if self.gradient_checkpointing:
                self.model.gradient_checkpointing_enable()

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model.resize_token_embeddings(len(self.tokenizer))

        self.model.to(self.device)

    def parameters(self):
        return self.model.parameters()

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def forward(self, **inputs):
        return self.model(**inputs)

    def prepare_generation_kwargs(self, generation_cfg: Dict) -> Dict:
        cfg = dict(generation_cfg)
        cfg.setdefault("max_new_tokens", 512)
        cfg.setdefault("do_sample", False)
        cfg.setdefault("temperature", 0.7)
        cfg.setdefault("top_p", 0.9)
        return cfg

    def generate(self, model_inputs: Dict, generation_cfg: Dict) -> torch.Tensor:
        kwargs = self.prepare_generation_kwargs(generation_cfg)
        input_ids = model_inputs["input_ids"].to(self.device)
        attention_mask = model_inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        pixel_values = model_inputs.get("pixel_values")
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.device)
        generation = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            **kwargs,
        )
        return generation
