import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
from io import BytesIO
import requests
import base64

class QwenV25Infer:
    def __init__(self, model=None, processor=None, api_endpoint=None, api_token=None):
        self.model = model
        self.processor = processor
        self.api_endpoint = api_endpoint
        self.api_token = api_token

    def infer(self, image_data, prompt):
        if self.api_endpoint and self.api_token:
            return self._infer_via_api(image_data, prompt)
        elif self.model and self.processor:
            return self._infer_locally(image_data, prompt)
        else:
            raise ValueError("Model and processor or API details must be provided for inference.")

    def _infer_locally(self, image_data, prompt):
        image = Image.open(BytesIO(image_data)).convert("RGB")
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text

    def _infer_via_api(self, image_data, prompt):
        headers = {"Authorization": f"Bearer {self.api_token}"}
        image_base64 = base64.b64encode(image_data).decode("utf-8")
        payload = {"inputs": {"image": image_base64, "text": prompt}}
        response = requests.post(self.api_endpoint, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API request failed with status code {response.status_code}"}
