# common/base_component.py
from abc import ABC
import logging

class BaseComponent(ABC):
    """
    Everyone gets a logger + config, but no forced methods.
    """
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug(f"{self.__class__.__name__} init with {self.config!r}")