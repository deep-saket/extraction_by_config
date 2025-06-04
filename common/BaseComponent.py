# common/base_component.py
from abc import ABC
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s │ %(message)s",
    datefmt="%H:%M:%S"
)

class BaseComponent(ABC):
    """
    Everyone gets a logger + config, but no forced methods.
    """
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug(f"{self.__class__.__name__} init with {self.config!r}")

        self.project_root = os.environ.get("PROJECT_ROOT")
        if not self.project_root:
            raise EnvironmentError("PROJECT_ROOT environment variable is not set.")

        print(f"Project Root: {self.project_root}")