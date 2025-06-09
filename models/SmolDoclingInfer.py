# file: components/smol_docling_component.py

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
from io import BytesIO
from huggingface_hub import InferenceClient

from docling_core.types.doc.document import DocTagsDocument   # type: ignore
from docling_core.types.doc import DoclingDocument             # type: ignore

from common import InferenceVLComponent

class SmolDoclingInfer(InferenceVLComponent):
    """
    SmolDocling-256M inference component (vision+text).
    """

    def __init__(self, config: dict = None):
        """
        config keys:
          - model_name (str, default "ds4sd/SmolDocling-256M-preview")
          - api_endpoint (str, optional)
          - api_token (str, optional)
          - device (str, optional, default "cuda")
        """
        super().__init__(config)
        self.model_name = self.config.get(
            "model_name", "ds4sd/SmolDocling-256M-preview"
        )
        self.api_endpoint = self.config.get("api_endpoint")
        self.api_token = self.config.get("api_token")
        self.device = self.config.get("device", "cuda")
        self.client = None
        self.model = None
        self.processor = None

        if self.api_endpoint and self.api_token:
            self.logger.info(f"Using SmolDocling API: {self.api_endpoint}")
            self.client = InferenceClient(model=self.api_endpoint, token=self.api_token)
        elif self.model_name:
            self.logger.info(f"Loading SmolDocling model locally: {self.model_name}")
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                _attn_implementation="flash_attention_2"
                if self.device == "cuda" else "eager"
            )
            self.model.eval()
            self.model.to(self.device)
            self.logger.info("SmolDocling model loaded successfully")
        else:
            raise ValueError("Either model_name or api_endpoint+api_token must be provided.")

    def infer(self, image_data=None, prompt: str = None) -> str:
        """
        Vision+Text inference: requires both image_data and prompt.
        """
        if image_data is None or prompt is None:
            raise ValueError("Both image_data and prompt are required for SmolDocling.")

        if self.client:
            self.logger.debug("Using remote SmolDocling API")
            return self._infer_via_api(image_data, prompt)
        elif self.model and self.processor:
            self.logger.debug("Performing local SmolDocling inference")
            return self._infer_locally(image_data, prompt)
        else:
            raise ValueError("SmolDocling component not properly initialized.")

    def _infer_locally(self, image_data, prompt: str) -> str:
        # Convert to PIL.Image
        if isinstance(image_data, bytes):
            image = Image.open(BytesIO(image_data)).convert("RGB")
        elif isinstance(image_data, Image.Image):
            image = image_data
        elif isinstance(image_data, str):
            image = Image.open(image_data).convert("RGB")
        else:
            raise ValueError("image_data must be bytes, PIL.Image, or file path.")

        # Build chat‐style prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self.processor(
            text=text,
            images=[image],
            return_tensors="pt"
        ).to(self.device)

        prompt_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=8192  # For long DocTags outputs
            )
        generated_ids = generated_ids[:, prompt_len:]
        doc_tags = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        self.logger.info("Local SmolDocling inference completed")
        return doc_tags

    def _infer_via_api(self, image_data, prompt: str) -> str:
        # Convert to PIL.Image
        if isinstance(image_data, bytes):
            image = Image.open(BytesIO(image_data)).convert("RGB")
        elif isinstance(image_data, Image.Image):
            image = image_data
        elif isinstance(image_data, str):
            image = Image.open(image_data).convert("RGB")
        else:
            raise ValueError("image_data must be bytes, PIL.Image, or file path.")

        response = self.client.text_to_image(prompt, image=image)
        if not response:
            self.logger.error("SmolDocling API failed")
            return ""
        self.logger.info("SmolDocling API inference completed")
        return response

    def parse_to_json(self, doc_tags: str) -> dict:
        """
        Convert raw DocTags → JSON dict via Docling Core.
        """
        tags_doc = DocTagsDocument.from_doctags_and_image_pairs([doc_tags], [None])
        doc = DoclingDocument.load_from_doctags(tags_doc, document_name="Document")
        return doc.export_to_dict()