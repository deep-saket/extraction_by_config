from typing import Optional

from src import AutoConfigGenerator

_generator_singleton: Optional[AutoConfigGenerator] = None


def get_auto_config_generator() -> AutoConfigGenerator:
    global _generator_singleton
    if _generator_singleton is None:
        _generator_singleton = AutoConfigGenerator()
    return _generator_singleton
