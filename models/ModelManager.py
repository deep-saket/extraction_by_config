import torch
import importlib
from common import BaseComponent
from config.loader import settings

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
    config = settings.get("model_manager", {})

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
                if class_name not in cls.config['models']:
                    raise KeyError(f"Expected config key '{class_name}' for local loading of '{class_name}'")
                model_name = cls.config['models'][class_name]["model_name_or_url"]
                try:
                    instance = ModelClass(model_name=model_name, device=device if device else torch.device(cls.config['models'][class_name].get("device", "cpu")))
                except Exception as e:
                    cls.logger.exception(f"Error instantiating {class_name}(model_name={model_name}, device={device})")
                    raise RuntimeError(f"Error instantiating {class_name}(model_name={model_name}, device={device})") from e

            else:  # model_loading == "api"
                api_endpoint = cls.config['models'][class_name].get("api_endpoint")
                api_token    = cls.config['models'][class_name].get("api_token")
                if api_token  or api_endpoint:
                    raise KeyError(f"Expected 'api_endpoint and 'api_token' in config for API loading of '{class_name}'")
                try:
                    instance = ModelClass(api_endpoint=api_endpoint, api_token=api_token)
                except Exception as e:
                    raise RuntimeError(f"Error instantiating {class_name}(api_endpoint={api_endpoint}, api_token=***)") from e

            # 5) Assign to class variable, e.g. ModelManager.QwenV25Infer or ModelManager.ColPaliInfer
            setattr(cls, class_name, instance)