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
    Everyone gets a class‐level logger + config, and subclasses
    automatically inherit a logger named after themselves.
    """
    # fallback logger for the base class itself
    logger: logging.Logger = logging.getLogger("BaseComponent")

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # each subclass gets its own logger
        cls.logger = logging.getLogger(cls.__name__)

    def __init__(self, config: dict = None):
        # if you still want an instance attribute, you can alias it here:
        self.config = config or {}
        self.logger.debug(f"{self.__class__.__name__} init with {self.config!r}")

        self.project_root = os.environ.get("PROJECT_ROOT")
        if not self.project_root:
            raise EnvironmentError("PROJECT_ROOT environment variable is not set.")

        self.logger.info(f"Project Root: {self.project_root}")