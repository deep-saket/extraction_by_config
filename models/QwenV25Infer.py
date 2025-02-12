import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
from io import BytesIO
from huggingface_hub import InferenceClient

class QwenV25Infer:
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

        if self.api_endpoint and self.api_token:
            self.client = InferenceClient(model=api_endpoint, token=api_token)
        elif model_name:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map="auto"
            ).to(self.device)
            self.processor = AutoProcessor.from_pretrained(model_name)
        else:
            raise ValueError("Either API details or a model name must be provided for inference.")

    def infer(self, image_data, prompt):
        """
        Performs inference on the provided image and prompt.

        Args:
            image_data (bytes): The image data in bytes format.
            prompt (str): The textual prompt for the model.

        Returns:
            str: The generated text from the model.

        Raises:
            ValueError: If the model and processor or API details are not properly initialized.
        """
        if self.client:
            return self._infer_via_api(image_data, prompt)
        elif self.model and self.processor:
            return self._infer_locally(image_data, prompt)
        else:
            raise ValueError("Model and processor or API details must be properly initialized for inference.")

    def _infer_locally(self, image_data, prompt):
        """
        Performs local inference using the loaded model.

        Args:
            image_data (bytes): The image data in bytes format.
            prompt (str): The textual prompt for the model.

        Returns:
            str: The generated text from the model.
        """
        image = Image.open(BytesIO(image_data)).convert("RGB")
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

        # Generate output
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text

    def _infer_via_api(self, image_data, prompt):
        """
        Performs inference via the specified API.

        Args:
            image_data (bytes): The image data in bytes format.
            prompt (str): The textual prompt for the model.

        Returns:
            dict: The API response containing the generated text or an error message.
        """
        image = Image.open(BytesIO(image_data)).convert("RGB")
        response = self.client.text_to_image(prompt, image=image)
        if response:
            return response
        else:
            return {"error": "API request failed."}
