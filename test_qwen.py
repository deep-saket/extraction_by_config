import torch
from transformers import ColPaliForRetrieval, ColPaliProcessor

class ColPaliInfer:
    def __init__(self, model_name_or_url):
        """
        Initializes the ColPaliInfer class by loading the model and processor.
        
        Args:
            model_name_or_url (str): The name or URL of the pretrained ColPali model.
        """
        # Load the ColPali model and processor
        self.model = ColPaliForRetrieval.from_pretrained(
            model_name_or_url,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        ).eval()
        self.processor = ColPaliProcessor.from_pretrained(model_name_or_url)

    def get_image_embedding(self, image):
        """
        Generates an embedding for the given image.
        
        Args:
            image (PIL.Image.Image): The input image.
        
        Returns:
            torch.Tensor: The image embedding.
        """
        # Process the image and get embeddings
        inputs = self.processor(images=image, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            embedding = self.model(**inputs).embeddings
        return embedding

    def get_text_embedding(self, text):
        """
        Generates an embedding for the given text query.
        
        Args:
            text (str): The input text query.
        
        Returns:
            torch.Tensor: The text embedding.
        """
        # Process the text and get embeddings
        inputs = self.processor(text=text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            embedding = self.model(**inputs).embeddings
        return embedding
