from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Iterable, List, Optional

import numpy as np
import torch
from PIL import Image
from torch import nn


class MockProcessor:
    """Lightweight processor that tokenizes JSON strings and downsamples images."""

    def __init__(self, vocab_size: int = 512, image_size: int = 32):
        self.vocab_size = vocab_size
        self.image_size = image_size
        self.pad_token_id = 0
        self.eos_token_id = 1

    def __call__(
        self,
        images: Optional[Iterable[Image.Image]] = None,
        text: Optional[List[str]] = None,
        truncation: bool = True,
        max_length: int = 256,
        return_tensors: str = "pt",
        **_: object,
    ) -> dict:
        if text is None:
            text = [""]

        batch_input_ids = []
        for item in text:
            token_ids = self._encode_text(item or "", max_length)
            batch_input_ids.append(token_ids)
        input_ids = torch.tensor(batch_input_ids, dtype=torch.long)
        attention_mask = (input_ids != self.pad_token_id).long()

        pixel_values = None
        if images is not None:
            processed_images = [self._encode_image(image) for image in images]
            pixel_values = torch.stack(processed_images)

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if pixel_values is not None:
            result["pixel_values"] = pixel_values
        return result

    def _encode_text(self, text: str, max_length: int) -> List[int]:
        max_len = max(2, max_length)
        tokens = [(ord(char) % (self.vocab_size - 2)) + 2 for char in text]
        tokens = tokens[: max_len - 1]
        tokens.append(self.eos_token_id)
        while len(tokens) < max_len:
            tokens.append(self.pad_token_id)
        return tokens

    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        img = image.convert("RGB").resize((self.image_size, self.image_size))
        array = np.asarray(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return tensor

    @contextmanager
    def as_target_processor(self):
        yield self


class MockVisionLanguageModel(nn.Module):
    """Tiny vision-language model used for smoke training runs."""

    def __init__(self, vocab_size: int = 512, hidden_size: int = 64, image_size: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.image_size = image_size

        self.token_embed = nn.Embedding(vocab_size, hidden_size)
        self.image_proj = nn.Linear(3 * image_size * image_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def gradient_checkpointing_enable(self):
        # No-op to align with Hugging Face models.
        return None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        **_: object,
    ) -> SimpleNamespace:
        token_emb = self.token_embed(input_ids)

        if pixel_values is None:
            batch_size = input_ids.shape[0]
            pixel_values = torch.zeros(
                (batch_size, 3, self.image_size, self.image_size), device=input_ids.device, dtype=token_emb.dtype
            )

        flat_pixels = pixel_values.reshape(pixel_values.size(0), -1)
        image_emb = self.image_proj(flat_pixels).unsqueeze(1)
        hidden = self.ln(token_emb + image_emb)
        logits = self.lm_head(hidden)
        return SimpleNamespace(logits=logits)


@dataclass
class MockComponentsConfig:
    vocab_size: int = 512
    hidden_size: int = 64
    image_size: int = 32
