from src import Parser
from host.shared.config import CONFIG_PATH

# Singleton parser instance initialized once
parser = Parser(CONFIG_PATH)