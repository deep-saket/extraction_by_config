import yaml
from models.QwenV25Infer import QwenV25Infer
from models.ColPaliInfer import ColPaliInfer

class ModelManager:
    qwen_infer = None
    colpali_infer = None
    config = None

    @classmethod
    def load_config(cls, config_path):
        with open(config_path, 'r') as file:
            cls.config = yaml.safe_load(file)

    @classmethod
    def initialize_models(cls):
        if cls.config is None:
            raise ValueError("Configuration not loaded. Call load_config first.")

        # Initialize QwenV25Infer
        model_loading = cls.config.get('model_loading', 'local')
        if model_loading == 'local':
            qwen_model_name_or_url = cls.config.get('qwen_model_name_or_url')
            qwen_infer = QwenV25Infer(model_name_or_url=qwen_model_name_or_url)
        elif model_loading == 'api':
            api_endpoint = cls.config.get('api_endpoint')
            api_token = cls.config.get('huggingface_api_token')
            qwen_infer = QwenV25Infer(api_endpoint=api_endpoint, api_token=api_token)
        else:
            raise ValueError("Invalid model_loading option in config.")
        cls.qwen_infer = qwen_infer

        # Initialize ColPaliInfer
        colpali_model_name_or_url = cls.config.get('colpali_model_name_or_url')
        cls.colpali_infer = ColPaliInfer(model_name_or_url=colpali_model_name_or_url)
