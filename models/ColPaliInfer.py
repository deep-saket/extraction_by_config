import torch
from transformers import AutoProcessor
from colpali_engine.models import ColQwen2, ColQwen2Processor
from PIL import Image
from io import BytesIO

class ColPaliInfer:
    """
    A class that provides inference capabilities for the ColPali model.
    
    This class loads a ColPali model and its corresponding processor, and
    offers methods to generate embeddings for both images and text queries.
    It also provides a method to retrieve the top-k candidates given a pre-computed query embedding.
    """
    def __init__(self, model_name="vidore/colqwen2-v0.1", device=None, torch_dtype=torch.bfloat16, device_map="auto"):
        """
        Initializes the ColPaliInfer class.

        Args:
            model_name (str): The name (or identifier) of the ColPali model to load.
            device (str or torch.device, optional): The device to run inference on 
                (e.g., "cuda" or "cpu"). If not specified, it defaults to "cuda" if available.
        """
        ## override device to cpu for colpali
        device = "cpu" if not device else device
        self.device = device

        # Save the model name.
        self.model_name = model_name
        
        # Load the ColPali model with the specified torch dtype.
        self.model = ColQwen2.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map  # Adjusts automatically; you can specify explicitly if needed.
        ).eval()
        self.model.to(self.device)  # Ensure model is on the correct device.
        
        # Load the corresponding processor.
        self.processor = ColQwen2Processor.from_pretrained(model_name)
        
    def get_image_embedding(self, image):
        """
        Generates an embedding for a given image.
        
        Args:
            image (PIL.Image.Image): The input image to embed.
        
        Returns:
            torch.Tensor: The embedding vector for the image.
        """
        # Process the image to prepare inputs for the model.
        batch_images = self.processor.process_images([image])
        # Move all tensors in the batch to the designated device.
        batch_images = {k: v.to(self.device) for k, v in batch_images.items()}
        
        # Generate embeddings without computing gradients.
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
        # Process the text to prepare inputs for the model.
        inputs = self.processor.process_queries([text] if isinstance(text, str) else text)
        # Move all tensors in the inputs to the designated device.
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate embeddings without computing gradients.
        with torch.no_grad():
            embedding = self.model(**inputs)
        return embedding