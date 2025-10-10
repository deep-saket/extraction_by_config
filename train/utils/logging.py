from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def create_logger(name: str, log_dir: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Create a namespaced logger with optional file output.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(Path(log_dir) / f"{name}.log")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger

