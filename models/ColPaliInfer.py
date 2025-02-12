import torch
from transformers import AutoProcessor
from colpali_engine.models import ColQwen2, ColQwen2Processor
from PIL import Image

class ColPaliInfer:
    """
    A class that provides inference capabilities for the ColPali model.
    
    This class handles loading a ColPali model and its processor, and provides
    methods to generate embeddings for both images and text.
    """
    def __init__(self, model_name="vidore/colqwen2-v0.1"):
        """
        Initializes the ColPaliInfer class.

        Args:
            model_name (str): The name of the ColPali model to use for adapter and processor.
            pretrained_model (str): The name of the pretrained ColPali model.
        """
        # Initialize the processor and model
        self.model_name = model_name
        self.model = ColQwen2.from_pretrained(
                            model_name,
                            torch_dtype=torch.bfloat16,
                            # device_map="cuda:0",  # or "mps" if on Apple Silicon
                        ).eval()
        self.processor = ColQwen2Processor.from_pretrained(model_name)
        
    def get_image_embedding(self, image):
        """
        Generates an embedding for a given image.

        Args:
            image (PIL.Image.Image): The input image to embed.

        Returns:
            torch.Tensor: The embedding vector for the image.
        """
        # Process the image and obtain its embedding
        batch_images = self.processor.process_images([image]).to(self.model.device)

        with torch.no_grad():
            embedding = self.model(**batch_images)
        return embedding

    def get_text_embedding(self, text):
        """
        Generates an embedding for a given text query.

        Args:
            text (str): The input text query.

        Returns:
            torch.Tensor: The embedding vector for the text.
        """
        # Process the text and obtain its embedding
        inputs = self.processor.process_queries([text]).to(self.model.device)
        with torch.no_grad():
            embedding = self.model(**inputs)
        return embedding
