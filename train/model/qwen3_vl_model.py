"""Wrapper around Qwen3-VL checkpoints for chat-style inference."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


class Qwen3VLModel:
    """Utility class for prompt/response style inference with Qwen3-VL."""

    def __init__(self, model_cfg: Dict):
        self.model_cfg = dict(model_cfg)
        self.model_name = self.model_cfg.get("pretrained_name", "Qwen/Qwen3-VL-4B-Instruct")

        dtype = self.model_cfg.get("dtype", "auto")
        if isinstance(dtype, str) and dtype != "auto":
            dtype = getattr(torch, dtype)

        model_kwargs = {
            "dtype": dtype,
            "device_map": self.model_cfg.get("device_map", "auto"),
        }
        attn_impl = self.model_cfg.get("attn_implementation")
        if attn_impl:
            model_kwargs["attn_implementation"] = attn_impl

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_name,
            **model_kwargs,
        )

        self.processor = AutoProcessor.from_pretrained(self.model_name)

    def to(self, device: torch.device | str) -> "Qwen3VLModel":
        self.model.to(device)
        return self

    def prepare_inputs(
        self,
        messages: List[Dict],
        add_generation_prompt: bool = True,
    ) -> Dict[str, torch.Tensor]:
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_dict=True,
            return_tensors="pt",
        )
        return inputs.to(self.model.device)

    def generate(
        self,
        messages: List[Dict],
        max_new_tokens: int = 256,
        **generate_kwargs,
    ) -> List[str]:
        inputs = self.prepare_inputs(messages)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            **generate_kwargs,
        )
        trimmed = [out[len(inp) :] for inp, out in zip(inputs["input_ids"], outputs)]
        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
