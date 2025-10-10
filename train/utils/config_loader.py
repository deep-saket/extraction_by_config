from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigLoader:
    """
    Simple YAML configuration loader.

    The entire configuration tree is parsed once during initialisation.
    Downstream consumers retrieve sections via `get` to avoid accidental
    mutation of the original mapping.
    """

    def __init__(self, config_path: str):
        path = Path(config_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            loaded: Dict[str, Any] = yaml.safe_load(handle) or {}
        self._config: Dict[str, Any] = loaded
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def get(self, *keys: str, default: Any = None) -> Any:
        """
        Retrieve a nested configuration value.

        Usage:
            loader.get("train", "batch_size", default=32)
        """
        node: Any = self._config
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def as_dict(self) -> Dict[str, Any]:
        """
        Return a shallow copy of the loaded configuration.
        """
        return dict(self._config)

