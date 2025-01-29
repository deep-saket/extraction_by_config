import yaml
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

class ModelManager:
    qwen_model = None
    qwen_processor = None
    colpali_model = None
    config = None

    @classmethod
    def load_config(cls, config_path):
        with open(config_path, 'r') as file:
            cls.config = yaml.safe_load(file)

    @classmethod
    def load_models(cls):
        if cls.config is None:
            raise ValueError("Configuration not loaded. Call load_config first.")

        if cls.config['model_loading'] == 'local':
            if cls.qwen_model is None or cls.qwen_processor is None:
                cls.qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    cls.config['model_name_or_url'], torch_dtype="auto", device_map="auto"
                )
                cls.qwen_processor = AutoProcessor.from_pretrained(cls.config['model_name_or_url'])
        elif cls.config['model_loading'] == 'api':
            if cls.qwen_processor is None:
                cls.qwen_processor = AutoProcessor.from_pretrained(cls.config['model_name_or_url'])
        else:
            raise ValueError("Invalid model_loading option in config.")

        # ColPali model loading logic should be added here
        # if cls.colpali_model is None:
        #     cls.colpali_model = ColPali()

        return cls.qwen_model, cls.qwen_processor, cls.colpali_model
