from typing import Optional
from src import Parser
from doc_ui.backend.shared.config import CONFIG_PATH

_parser_singleton: Optional[Parser] = None

def get_parser() -> Parser:
    global _parser_singleton
    if _parser_singleton is None:
        _parser_singleton = Parser(CONFIG_PATH)
    return _parser_singleton
