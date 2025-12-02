import base64
import json
from io import BytesIO
from typing import Optional

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from common import InferenceVLComponent


class Qwen3VLInfer(InferenceVLComponent):
    """Inference helper for Qwen3-VL models with local or API backends."""

    def __init__(self, model_name: Optional[str] = None, api_endpoint: Optional[str] = None,
                 api_token: Optional[str] = None, device: str = "cuda"):
        self.api_endpoint = api_endpoint
        self.api_token = api_token
        self.device = device
        self.client = None
        self.model = None
        self.processor = None

        if self.api_endpoint and self.api_token:
            self.client = True
        elif model_name:
            print(f"Loading {model_name} model...")
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                dtype="auto",
                device_map="auto" if device != "cpu" else None,
            ).to(self.device)
            print("Model loaded!")
            self.processor = AutoProcessor.from_pretrained(model_name)
        else:
            raise ValueError("Either API details or a model name must be provided for inference.")

    def infer(self, image_data, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        if not image_data:
            raise ValueError("Image data cannot be None")
        if not prompt:
            raise ValueError("Prompt must be provided")

        if self.client:
            return self._infer_via_api(image_data, prompt, max_new_tokens=max_new_tokens)
        if self.model and self.processor:
            return self._infer_locally(image_data, prompt, max_new_tokens=max_new_tokens)
        raise ValueError("Model and processor or API details must be properly initialized for inference.")

    def _infer_locally(self, image_data, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        image = self._to_pil(image_data)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        max_tokens = max_new_tokens or 500
        generated = self.model.generate(**inputs, max_new_tokens=max_tokens)
        trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated)]
        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def infer_lang(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        if not prompt:
            raise ValueError("Prompt must be provided")

        if self.client:
            return self._infer_lang_via_api(prompt, max_new_tokens=max_new_tokens)

        if not (self.model and self.processor):
            raise ValueError("Model and processor must be initialized for local inference.")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        max_tokens = max_new_tokens or 500
        generated = self.model.generate(**inputs, max_new_tokens=max_tokens)
        trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated)]
        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def _infer_via_api(self, image_data, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        data_url = self._image_to_data_url(image_data)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        payload = {
            "model": "auto",
            "messages": messages,
            "temperature": 0,
        }
        if max_new_tokens is not None:
            payload["max_tokens"] = int(max_new_tokens)
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }
        import requests

        resp = requests.post(
            self.api_endpoint.rstrip("/") + "/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        try:
            content = data["choices"][0]["message"]["content"]
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
                        return part["text"]
                return json.dumps(content)
            return content
        except Exception:
            return json.dumps(data)

    def _infer_lang_via_api(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]
        payload = {
            "model": "auto",
            "messages": messages,
            "temperature": 0,
        }
        if max_new_tokens is not None:
            payload["max_tokens"] = int(max_new_tokens)
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }
        import requests

        resp = requests.post(
            self.api_endpoint.rstrip("/") + "/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        try:
            content = data["choices"][0]["message"]["content"]
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
                        return part["text"]
                return json.dumps(content)
            return content
        except Exception:
            return json.dumps(data)

    @staticmethod
    def _to_pil(image_data) -> Image.Image:
        if isinstance(image_data, Image.Image):
            return image_data.convert("RGB")
        if isinstance(image_data, bytes):
            return Image.open(BytesIO(image_data)).convert("RGB")
        if isinstance(image_data, str):
            return Image.open(image_data).convert("RGB")
        raise ValueError("Unsupported image format")

    @staticmethod
    def _image_to_data_url(image_data) -> str:
        image = Qwen3VLInfer._to_pil(image_data)
        buf = BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"
