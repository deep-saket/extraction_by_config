import yaml
import torch
import importlib
from common import BaseComponent


class ModelManager(BaseComponent):
    """
    Dynamically load and instantiate model classes given their class-name strings.
    All models respect a global `model_loading` setting from config:
      - If model_loading == "local": instantiate with (model_name=<…>, device=<…>)
      - If model_loading == "api": instantiate with (api_endpoint=<…>, api_token=<…>)
    Assumes each class resides in a module named models.<ClassName> and that
    config contains the appropriate keys for each mode:
      - <ClassName>_model_name_or_url
      - <ClassName>_api_endpoint
      - <ClassName>_api_token
    """
    config = None

    @classmethod
    def load_config(cls, config_path: str):
        """
        Load YAML configuration. Expected keys include:
          model_loading: "local" or "api"
          <ClassName>_model_name_or_url: string
          <ClassName>_api_endpoint: string
          <ClassName>_api_token: string
        """
        with open(config_path, "r") as f:
            cls.config = yaml.safe_load(f)

    @classmethod
    def initialize_models(
        cls,
        device: torch.device = torch.device("cpu"),
        model_classes: list[str] = []
    ):
        """
        For each class_name in model_classes:
          1) If cls already has a non-None attribute named class_name, skip instantiation.
          2) Import module "models.<class_name>"
          3) Retrieve class "<class_name>"
          4) If config["model_loading"] == "local":
               – Look up "<class_name>_model_name_or_url"
               – Instantiate: ModelClass(model_name=<value>, device=device)
             Else if "api":
               – Look up "<class_name>_api_endpoint" and "<class_name>_api_token"
               – Instantiate: ModelClass(api_endpoint=<…>, api_token=<…>)
             Else:
               – Raise ValueError
          5) Assign the instance to cls.<class_name>
        """
        if cls.config is None:
            raise ValueError("Configuration not loaded. Call load_config first.")

        # Determine global loading mode
        model_loading = cls.config.get("model_loading", "local")
        if model_loading not in ("local", "api"):
            raise ValueError(f"Invalid model_loading: {model_loading}. Must be 'local' or 'api'.")

        for class_name in model_classes:
            # 1) If already instantiated, skip
            existing = getattr(cls, class_name, None)
            if existing is not None:
                continue

            # 2) Dynamically import the module "models.<ClassName>"
            module_name = f"models.{class_name}"
            try:
                module = importlib.import_module(module_name)
            except ModuleNotFoundError as e:
                raise ImportError(f"Could not import module '{module_name}' for class '{class_name}'") from e

            # 3) Retrieve the class object
            try:
                ModelClass = getattr(module, class_name)
            except AttributeError as e:
                raise ImportError(f"Module '{module_name}' does not define class '{class_name}'") from e

            # 4) Instantiate based on loading mode
            if model_loading == "local":
                config_key = f"{class_name}_model_name_or_url"
                if config_key not in cls.config:
                    raise KeyError(f"Expected config key '{config_key}' for local loading of '{class_name}'")
                model_name = cls.config[config_key]
                try:
                    instance = ModelClass(model_name=model_name, device=device)
                except Exception as e:
                    cls.logger.exception(f"Error instantiating {class_name}(model_name={model_name}, device={device})")
                    raise RuntimeError(f"Error instantiating {class_name}(model_name={model_name}, device={device})") from e

            else:  # model_loading == "api"
                endpoint_key = f"{class_name}_api_endpoint"
                token_key    = f"{class_name}_api_token"
                if endpoint_key not in cls.config or token_key not in cls.config:
                    raise KeyError(f"Expected '{endpoint_key}' and '{token_key}' in config for API loading of '{class_name}'")
                api_endpoint = cls.config[endpoint_key]
                api_token    = cls.config[token_key]
                try:
                    instance = ModelClass(api_endpoint=api_endpoint, api_token=api_token)
                except Exception as e:
                    raise RuntimeError(f"Error instantiating {class_name}(api_endpoint={api_endpoint}, api_token=***)") from e

            # 5) Assign to class variable, e.g. ModelManager.QwenV25Infer or ModelManager.ColPaliInfer
            setattr(cls, class_name, instance)