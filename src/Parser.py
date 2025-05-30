import json
from vector_retrieve import PDFProcessor
from src.helper import PromptBuilder, VLMProcessor, PageFinder
from models import ModelManager
from common import ExtractionState
import torch


class Parser:
    def __init__(self, extraction_config, config_path="config/settings.yml", device=torch.device('mps')):
        # Initialize shared models
        ModelManager.load_config(config_path)
        ModelManager.initialize_models(device)

        # Set up components
        self.pdf_processor    = PDFProcessor(ModelManager.colpali_infer)
        self.extraction_config = extraction_config
        self.prompt_builder   = PromptBuilder()
        self.vlm_processor    = VLMProcessor(ModelManager.qwen_infer)
        self.page_finder      = PageFinder(self.pdf_processor)

    def perform_de(self, pdf_path, output_json_path):
        # Reset state at start of cycle
        ExtractionState.reset()

        # Populate global state via PDFProcessor
        self.pdf_processor(pdf_path)

        extracted_data = {}

        for field in self.extraction_config:
            name = field["field_name"]
            desc = field["description"]
            typ  = field["type"]
            pages = field.get("probable_pages") or self.page_finder(ExtractionState.embeddings, name, desc)

            # extract using only global images and embeddings
            values = self._process_specified_pages(name, desc, pages, typ)
            extracted_data[name] = values[0] if typ=="key-value" and values else values

        self._create_json(extracted_data, output_json_path)

    def _process_specified_pages(self, field_name, description, pages, extraction_type):
        field_values = []
        images = ExtractionState.get_images()
        for num, img_path in images:
            if num in pages:
                prompt = self.prompt_builder(field_name, description, num, extraction_type)
                from PIL import Image
                img = Image.open(img_path).convert("RGB")
                content = self.vlm_processor(img, prompt)
                field_values.append(content)
        return field_values

    def _create_json(self, data, path):
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
