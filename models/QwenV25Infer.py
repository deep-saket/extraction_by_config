from typing import Optional

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
from io import BytesIO
import base64
import json
import requests
from common import InferenceVLComponent
from abc import abstractmethod


class QwenV25Infer(InferenceVLComponent):
    """
    A class to perform inference using the Qwen2.5-VL model, either locally or via an API.

    Attributes:
        model_name (str): The name of the pretrained model to load.
        api_endpoint (str): The API endpoint for inference.
        api_token (str): The API token for authentication.
        device (str): The device to run the model on ('cuda' or 'cpu').
        model (Qwen2_5_VLForConditionalGeneration): The loaded Qwen2.5-VL model.
        processor (AutoProcessor): The processor for preparing inputs for the model.
        client (InferenceClient): The client for making API requests.
    """

    def __init__(self, model_name=None, api_endpoint=None, api_token=None, device='cuda'):
        """
        Initializes the QwenV25Infer class.

        Args:
            model_name (str, optional): The name of the pretrained model to load.
            api_endpoint (str, optional): The API endpoint for inference.
            api_token (str, optional): The API token for authentication.
            device (str, optional): The device to run the model on ('cuda' or 'cpu'). Defaults to 'cuda'.

        Raises:
            ValueError: If neither API details nor a model name are provided.
        """
        self.api_endpoint = api_endpoint
        self.api_token = api_token
        self.device = device
        self.client = None
        self.model = None
        self.processor = None
        self.default_max_new_tokens = 2048

        if self.api_endpoint and self.api_token:
            # Use OpenAI-compatible HTTP interface
            self.client = True
        elif model_name:
            print(f"Loading {model_name} model...")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name #, torch_dtype=torch.float16, device_map="auto"
            ).to(self.device)
            print(f"Model loaded!")
            self.processor = AutoProcessor.from_pretrained(model_name)
        else:
            raise ValueError("Either API details or a model name must be provided for inference.")

    def infer(self, image_data, prompt, max_new_tokens: Optional[int] = None):
        """
        Performs inference on the provided image and prompt.

        Args:
            image_data (Union[bytes, str, Image.Image]): The image data as bytes, file path, or PIL Image.
            prompt (str): The textual prompt for the model.

        Returns:
            str: The generated text from the model.

        Raises:
            ValueError: If the model and processor or API details are not properly initialized,
                       or if input parameters are invalid.
        """
        if not image_data:
            raise ValueError("Image data cannot be None")
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string")

        try:
            if self.client:
                response = self._infer_via_api(image_data, prompt, max_new_tokens=max_new_tokens)
                return response if isinstance(response, str) else str(response)
            elif self.model and self.processor:
                return self._infer_locally(image_data, prompt, max_new_tokens=max_new_tokens)
            else:
                raise ValueError("Model and processor or API details must be properly initialized for inference.")
        except Exception as e:
            raise RuntimeError(f"Inference failed: {str(e)}") from e

    def _infer_locally(self, image_data, prompt, max_new_tokens: Optional[int] = None):
        """
        Performs local inference using the loaded model.

        Args:
            image_data (bytes): The image data in bytes format.
            prompt (str): The textual prompt for the model.

        Returns:
            str: The generated text from the model.
        """
        if isinstance(image_data, bytes):
            image = Image.open(BytesIO(image_data)).convert("RGB")
        elif isinstance(image_data, Image.Image):
            image = image_data
        elif isinstance(image_data, str):
            image = Image.open(image_data).convert("RGB")
        else:
            raise ValueError("Image must be either bytes or Image object.")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Prepare inputs using the processor and process_vision_info
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # Record how many tokens the prompt took:
        prompt_len = inputs["input_ids"].shape[-1]

        # Generate output
        max_tokens = max_new_tokens or self.default_max_new_tokens
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
        generated_ids = generated_ids[:, prompt_len:]
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text

    def _infer_via_api(self, image_data, prompt, max_new_tokens: Optional[int] = None):
        """Call an OpenAI-compatible chat endpoint and return assistant text."""
        if isinstance(image_data, bytes):
            img = Image.open(BytesIO(image_data)).convert("RGB")
        elif isinstance(image_data, Image.Image):
            img = image_data
        elif isinstance(image_data, str):
            img = Image.open(image_data).convert("RGB")
        else:
            raise ValueError("Image must be either bytes, PIL Image, or path.")

        # Encode image as base64 data URL (PNG)
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        data_url = f"data:image/png;base64,{b64}"

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
            "model": "auto",  # server may ignore/route
            "messages": messages,
            "temperature": 0,
        }
        if max_new_tokens is not None:
            payload["max_tokens"] = int(max_new_tokens)
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }
        url = self.api_endpoint.rstrip('/') + "/v1/chat/completions"
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
        resp.raise_for_status()
        data = resp.json()
        # Parse OpenAI-style response
        try:
            msg = data["choices"][0]["message"]["content"]
            if isinstance(msg, list):
                # content may be list of parts
                for part in msg:
                    if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
                        return part["text"]
                return json.dumps(msg)
            return msg
        except Exception:
            # fallback
            return json.dumps(data)

    def infer_lang(self, prompt: str = None, max_new_tokens: Optional[int] = None) -> str:
        """
        Run inference using only text input.

        Args:
            prompt: Optional; textual prompt for language-based inference.

        Returns:
            str: The model's generated text response.
        """
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string")

        try:
            if self.client:
                # OpenAI chat completion text-only
                headers = {
                    "Authorization": f"Bearer {self.api_token}",
                    "Content-Type": "application/json",
                }
                url = self.api_endpoint.rstrip('/') + "/v1/chat/completions"
                payload = {
                    "model": "auto",
                    "messages": [
                        {"role": "user", "content": [{"type": "text", "text": prompt}]}
                    ],
                    "temperature": 0,
                }
                if max_new_tokens is not None:
                    payload["max_tokens"] = int(max_new_tokens)
                r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
                r.raise_for_status()
                data = r.json()
                try:
                    msg = data["choices"][0]["message"]["content"]
                    if isinstance(msg, list):
                        for part in msg:
                            if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
                                return part["text"]
                        return json.dumps(msg)
                    return msg
                except Exception:
                    return json.dumps(data)
            elif self.model and self.processor:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]

                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(text=[text], return_tensors="pt").to(self.device)

                # Record how many tokens the prompt took:
                prompt_len = inputs["input_ids"].shape[-1]

                # Generate output
                max_tokens = max_new_tokens or self.default_max_new_tokens
                with torch.no_grad():
                    generated_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
                generated_ids = generated_ids[:, prompt_len:]

                return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            else:
                raise ValueError("Model and processor or API details must be properly initialized for inference.")
        except Exception as e:
            raise RuntimeError(f"Text inference failed: {str(e)}") from e
