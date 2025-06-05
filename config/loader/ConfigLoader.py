import yaml
from pathlib import Path
from typing import Any, Dict

class ConfigLoader:
    """
    Multiton loader: you get one instance per unique config_path.
    If you call ConfigLoader() again with the same path, it returns
    the same instance (and same loaded config). Different paths → new instances.
    """
    _instances: Dict[str, "ConfigLoader"] = {}

    def __new__(cls, config_path: str):
        # Normalize and resolve the path so that logically equivalent
        # paths map to the same key.
        normalized = str(Path(config_path).expanduser().resolve())

        if normalized not in cls._instances:
            # Create, load, and cache a new instance for this path
            instance = super().__new__(cls)
            instance._load(normalized)
            cls._instances[normalized] = instance

        return cls._instances[normalized]

    def _load(self, config_path: str):
        path = Path(config_path)
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with path.open("r") as f:
            self._config: Dict[str, Any] = yaml.safe_load(f)

    def get_config(self) -> Dict[str, Any]:
        return self._config