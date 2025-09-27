import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
from PIL import Image
import base64
import json
import requests
from common import InferenceVLComponent
from io import BytesIO
from tempfile import NamedTemporaryFile


class H2OVLInfer(InferenceVLComponent):
    """
    Inference class for H2O VL Mississippi-800M model.
    """

    def __init__(self, model_name=None, api_endpoint=None, api_token=None, device='cuda'):
        self.api_endpoint = api_endpoint
        self.api_token = api_token
        self.device = device
        self.client = None
        self.model = None
        self.tokenizer = None
        self.generation_config = dict(max_new_tokens=2048, do_sample=True)

        if self.api_endpoint and self.api_token:
            # OpenAI-compatible remote
            self.client = True
        elif model_name:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            config.llm_config._attn_implementation = 'flash_attention_2' if self.device == 'cuda' else 'eager'

            self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                config=config,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).eval().to(self.device)

            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
        else:
            raise ValueError("Provide either API details or a model name for inference.")

    def infer(self, image_data, prompt):
        """
        Run inference on image + text prompt.
        """
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string")

        try:
            if self.client:
                return self._infer_via_api(image_data, prompt)
            elif self.model and self.tokenizer:
                return self._infer_locally(image_data, prompt)
            else:
                raise ValueError("Initialization incomplete for inference.")
        except Exception as e:
            self.logger.exception(f"Inference failed: {str(e)}")
            raise RuntimeError(f"Inference failed: {str(e)}") from e

    def _infer_locally(self, image_data, prompt):
        """
        Local inference via model.chat(). Accepts image path or bytes or PIL.Image.
        """
        if isinstance(image_data, bytes):
            with NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                img = Image.open(BytesIO(image_data)).convert("RGB")
                img.save(tmp.name)
                image_data = tmp.name

        elif isinstance(image_data, Image.Image):
            with NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                image_data.save(tmp.name)
                image_data = tmp.name

        history = None
        response, history = self.model.chat(
            self.tokenizer,
            image_data,
            prompt,
            self.generation_config,
            history=history,
            return_history=True
        )
        return response

    def infer_lang(self, prompt: str = None) -> str:
        """
        Text-only inference.
        """
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string")

        try:
            if self.client:
                headers = {"Authorization": f"Bearer {self.api_token}", "Content-Type": "application/json"}
                payload = {
                    "model": "auto",
                    "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                    "temperature": 0,
                }
                url = self.api_endpoint.rstrip('/') + "/v1/chat/completions"
                r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
                r.raise_for_status()
                data = r.json()
                try:
                    content = data["choices"][0]["message"]["content"]
                    if isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                return part.get("text", "")
                        return json.dumps(content)
                    return content
                except Exception:
                    return json.dumps(data)
            elif self.model and self.tokenizer:
                history = None
                response, history = self.model.chat(
                    self.tokenizer,
                    None,
                    prompt,
                    self.generation_config,
                    history=history,
                    return_history=True
                )
                return response
            else:
                raise ValueError("Initialization incomplete for inference.")
        except Exception as e:
            raise RuntimeError(f"Text inference failed: {str(e)}") from e

    def _infer_via_api(self, image_data, prompt):
        """
        OpenAI-compatible chat completion with image.
        """
        # Prepare image
        if isinstance(image_data, bytes):
            img = Image.open(BytesIO(image_data)).convert("RGB")
        elif isinstance(image_data, Image.Image):
            img = image_data
        elif isinstance(image_data, str):
            img = Image.open(image_data).convert("RGB")
        else:
            raise ValueError("image_data must be bytes, PIL.Image, or path")

        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        data_url = f"data:image/png;base64,{b64}"

        messages = [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": prompt},
            ]}
        ]
        payload = {"model": "auto", "messages": messages, "temperature": 0}
        headers = {"Authorization": f"Bearer {self.api_token}", "Content-Type": "application/json"}
        url = self.api_endpoint.rstrip('/') + "/v1/chat/completions"
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
        r.raise_for_status()
        data = r.json()
        try:
            content = data["choices"][0]["message"]["content"]
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        return part.get("text", "")
                return json.dumps(content)
            return content
        except Exception:
            return json.dumps(data)
