import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
from io import BytesIO
from huggingface_hub import InferenceClient

class QwenV25Infer:
    def __init__(self, model_name=None, api_endpoint=None, api_token=None, device='cuda'):
        self.api_endpoint = api_endpoint
        self.api_token = api_token
        self.client = None
        self.device = device

        if self.api_endpoint and self.api_token:
            self.client = InferenceClient(model=api_endpoint, token=api_token)
        elif model_name:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch.float16
            ).to(self.device)
            self.processor = AutoProcessor.from_pretrained(model_name)
        else:
            raise ValueError("Either API details or a model name must be provided for inference.")

    def infer(self, image_data, prompt):
        if self.client:
            return self._infer_via_api(image_data, prompt)
        elif hasattr(self, 'model') and hasattr(self, 'processor'):
            return self._infer_locally(image_data, prompt)
        else:
            raise ValueError("Model and processor or API details must be properly initialized for inference.")

    def _infer_locally(self, image_data, prompt):
        image = Image.open(BytesIO(image_data)).convert("RGB")
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text

    def _infer_via_api(self, image_data, prompt):
        image = Image.open(BytesIO(image_data)).convert("RGB")
        response = self.client.text_to_image(prompt, image=image)
        if response:
            return response
        else:
            return {"error": "API request failed."}
