import torch
from colpali_engine.models import ColQwen2, ColQwen2Processor

class ColPaliInfer:
    def __init__(self, model_name_or_url):
        self.model = ColQwen2.from_pretrained(
            model_name_or_url,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        self.processor = ColQwen2Processor.from_pretrained(model_name_or_url)

    def get_image_embedding(self, image):
        inputs = self.processor.process_images([image]).to(self.model.device)
        with torch.no_grad():
            embedding = self.model(**inputs)
        return embedding

    def get_text_embedding(self, text):
        inputs = self.processor.process_queries([text]).to(self.model.device)
        with torch.no_grad():
            embedding = self.model(**inputs)
        return embedding
