import json
import base64
import requests
from io import BytesIO
from PIL import Image
import torch
from transformers import AutoProcessor
from models.model_manager import ModelManager

class VLInference:
    def __init__(self):
        self.qwen_model, self.qwen_processor, _ = ModelManager.load_models()
        self.config = ModelManager.config

    def infer(self, image_data, prompt):
        if self.config['model_loading'] == 'local':
            return self._infer_local(image_data, prompt)
        elif self.config['model_loading'] == 'api':
            return self._infer_api(image_data, prompt)
        else:
            raise ValueError("Invalid model_loading option in config.")

    def _infer_local(self, image_data, prompt):
        image = Image.open(BytesIO(image_data))
        inputs = self.qwen_processor(images=image, text=prompt, return_tensors="pt").to(self.qwen_model.device)
        with torch.no_grad():
            generated_ids = self.qwen_model.generate(**inputs, max_new_tokens=512)
        generated_text = self.qwen_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        try:
            return json.loads(generated_text)
        except json.JSONDecodeError:
            return {"error": "Failed to parse JSON output"}

    def _infer_api(self, image_data, prompt):
        api_endpoint = self.config['api_endpoint']
        api_token = self.config['huggingface_api_token']
        headers = {"Authorization": f"Bearer {api_token}"}
        image_base64 = base64.b64encode(image_data).decode("utf-8")
        payload = {"inputs": {"image": image_base64, "text": prompt}}
        response = requests.post(api_endpoint, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API request failed with status code {response.status_code}"}
