import json
import torch
from transformers import PretrainedConfig
from colpali_engine.models import ColQwen2, ColQwen2Processor
from common import InferenceVLComponent
from PIL import Image
from io import BytesIO

class ColPaliInfer(InferenceVLComponent):
    """
    Inference wrapper for ColQwen2 that ensures decoder_config is a proper config,
    and logs each step via the inherited logger.
    """
    def __init__(
        self,
        model_name="vidore/colqwen2-v0.1",
        device=None,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ):
        super().__init__()  # initializes self.logger, etc.
        device = "cpu" if not device else device
        self.logger.info(f"Initializing ColPaliInfer: model={model_name}, device={device}")

        # 1) Load raw config
        self.logger.debug("Loading raw config from pretrained")
        config = ColQwen2.config_class.from_pretrained(model_name)

        # 2) Wrap decoder_config if it's a dict (only if attribute exists)
        decoder_cfg = getattr(config, "decoder_config", None)
        if isinstance(decoder_cfg, dict):
            self.logger.warning("decoder_config is plain dict; wrapping into PretrainedConfig")
            config.decoder_config = PretrainedConfig.from_dict(decoder_cfg)

        # 3) Load model with the corrected config
        self.logger.info("Loading ColQwen2 model from pretrained")
        self.model = (
            ColQwen2.from_pretrained(
                model_name,
                #config=config,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
            .eval()
            .to(device)
        )
        self.logger.info("Model loaded and moved to %s", device)

        # 4) Load processor
        self.logger.info("Loading ColQwen2Processor")
        self.processor = ColQwen2Processor.from_pretrained(model_name)

        self.device = device
        self.model_name = model_name
        self.logger.info("ColPaliInfer initialization complete")

    def get_image_embedding(self, image: Image.Image) -> torch.Tensor:
        self.logger.debug("Generating image embedding")
        batch = self.processor.process_images([image])
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            emb = self.model(**batch)
        self.logger.debug("Image embedding generated")
        return emb

    def get_text_embedding(self, text: str) -> torch.Tensor:
        self.logger.debug("Generating text embedding for %r", text)
        inputs = self.processor.process_queries([text] if isinstance(text, str) else text)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            emb = self.model(**inputs)
        self.logger.debug("Text embedding generated")
        return emb

    def infer(self, image_data=None, prompt: str = None) -> str:
        """
        Run inference on the provided inputs:
          - If image_data is given, returns its embedding.
          - If prompt is given, returns its embedding.
        Exactly one of image_data or prompt must be provided.
        """
        if image_data is not None and prompt is None:
            self.logger.info("Running inference on image_data")
            # Normalize to PIL Image
            if isinstance(image_data, bytes):
                image = Image.open(BytesIO(image_data)).convert("RGB")
            elif isinstance(image_data, str):
                image = Image.open(image_data).convert("RGB")
            elif isinstance(image_data, Image.Image):
                image = image_data
            else:
                raise TypeError(f"Unsupported image_data type: {type(image_data)}")
            emb = self.get_image_embedding(image)

        elif prompt is not None and image_data is None:
            self.logger.info("Running inference on prompt")
            emb = self.get_text_embedding(prompt)

        else:
            raise ValueError("Provide exactly one of 'image_data' or 'prompt' to infer()")

        # Convert tensor to list and serialize as JSON
        emb_list = emb.squeeze().cpu().tolist()
        result = json.dumps(emb_list)
        self.logger.debug("Inference result serialized to JSON")
        return result