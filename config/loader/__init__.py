import os
import dotenv
from config.loader.ConfigLoader import ConfigLoader

dotenv.load_dotenv()

project_root = os.environ.get("PROJECT_ROOT")
settings = ConfigLoader(os.path.join(project_root, "config/files/settings.yml")).get_config()
prompts = ConfigLoader(os.path.join(project_root, "config/files/prompts.yml")).get_config()

# Load additional prompts (retry prompts, etc.) with fallback to the older misspelled file
_additional_path = os.path.join(project_root, "config/files/additional_prompts.yml")
if not os.path.exists(_additional_path):
    legacy = os.path.join(project_root, "config/files/addtional_prompts.yml")
    if os.path.exists(legacy):
        _additional_path = legacy

try:
    additional_prompts = ConfigLoader(_additional_path).get_config() if os.path.exists(_additional_path) else {}
except Exception:
    additional_prompts = {}
