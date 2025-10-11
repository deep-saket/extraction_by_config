import base64
import json
import os
from io import BytesIO
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import requests
from PIL import Image

from common import InferenceVLComponent


class AzureOpenAIInfer(InferenceVLComponent):
    """
    Inference component that talks to an Azure OpenAI chat completion endpoint.

    Expected config:
        - api_endpoint: Either the full chat completion URL or the Azure resource base URL.
        - api_token: Azure OpenAI API key.
        - Optional extras via kwargs/env:
            * deployment_name / AZURE_OPENAI_DEPLOYMENT
            * api_version / AZURE_OPENAI_API_VERSION
    """

    def __init__(self, api_endpoint=None, api_token=None, deployment_name=None, api_version=None, **kwargs):
        if not api_endpoint or not api_token:
            raise ValueError("AzureOpenAIInfer requires api_endpoint and api_token.")

        self.api_token = api_token
        self.raw_endpoint = api_endpoint.strip()
        self.api_version = api_version or kwargs.get("api_version") or os.getenv("AZURE_OPENAI_API_VERSION")
        self.deployment_name = deployment_name or kwargs.get("deployment_name") or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        self.request_url = self._build_request_url()
        self.headers = {
            "api-key": self.api_token,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def infer(self, image_data, prompt):
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string")

        if image_data is None:
            raise ValueError("Image data is required for vision inference")

        try:
            image_part = self._encode_image_content(image_data)
            messages = [{
                "role": "user",
                "content": [image_part, {"type": "text", "text": prompt}],
            }]
            return self._invoke_chat(messages)
        except Exception as exc:
            raise RuntimeError(f"Azure OpenAI vision inference failed: {exc}") from exc

    def infer_lang(self, prompt: str = None) -> str:
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string")

        try:
            messages = [{
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }]
            return self._invoke_chat(messages)
        except Exception as exc:
            raise RuntimeError(f"Azure OpenAI text inference failed: {exc}") from exc

    def _encode_image_content(self, image_data):
        if isinstance(image_data, bytes):
            image = Image.open(BytesIO(image_data)).convert("RGB")
        elif isinstance(image_data, Image.Image):
            image = image_data.convert("RGB")
        elif isinstance(image_data, str):
            image = Image.open(image_data).convert("RGB")
        else:
            raise ValueError("Image must be bytes, PIL.Image, or a file path.")

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}

    def _invoke_chat(self, messages):
        payload = {
            "messages": messages,
            "temperature": 0,
        }
        response = requests.post(self.request_url, headers=self.headers, data=json.dumps(payload), timeout=120)
        response.raise_for_status()
        data = response.json()
        return self._extract_text(data)

    def _extract_text(self, data):
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"Unexpected Azure OpenAI response shape: {data}") from exc

        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in {"text", "output_text"}:
                    text = part.get("text")
                    if text:
                        return text
        elif isinstance(content, str):
            return content

        # Fall back to stringified payload
        return json.dumps(data)

    def _build_request_url(self):
        parsed = urlparse(self.raw_endpoint)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("api_endpoint must be a valid URL.")

        query = parse_qs(parsed.query)
        version = self.api_version or (query.get("api-version", [None])[0])

        if "/deployments/" in parsed.path:
            query["api-version"] = [version] if version else query.get("api-version", [])
            if not query.get("api-version"):
                raise ValueError("Azure OpenAI api-version is required.")
            flattened = {key: vals if len(vals) > 1 else vals[0] for key, vals in query.items()}
            new_query = urlencode(flattened, doseq=True)
            return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment))

        if not self.deployment_name:
            raise ValueError("Azure OpenAI deployment_name must be provided.")
        if not version:
            raise ValueError("Azure OpenAI api_version must be provided.")

        base_path = parsed.path.rstrip("/")
        full_path = f"{base_path}/openai/deployments/{self.deployment_name}/chat/completions"
        return urlunparse(
            (parsed.scheme, parsed.netloc, full_path, parsed.params, urlencode({"api-version": version}), parsed.fragment)
        )
