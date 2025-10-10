import os
import dotenv
from config.loader.ConfigLoader import ConfigLoader
from pathlib import Path

dotenv.load_dotenv()

# Prefer PROJECT_ROOT env var, otherwise fallback to repo root (two parents above this file)
project_root = os.environ.get("PROJECT_ROOT")
if not project_root:
    project_root = str(Path(__file__).resolve().parents[2])

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

# --- NEW: load hosting config as an additional, non-destructive variable ---
_hosting_path = os.path.join(project_root, "config/files/hosting.yml")
try:
    hosting = ConfigLoader(_hosting_path).get_config() if os.path.exists(_hosting_path) else {}
except Exception:
    hosting = {}

# Auto-config prompt templates (used when generating configs from PDFs)
_auto_config_prompts_path = os.path.join(project_root, "config/files/auto_config_prompts.yml")
try:
    auto_config_prompts = ConfigLoader(_auto_config_prompts_path).get_config() if os.path.exists(_auto_config_prompts_path) else {}
except Exception:
    auto_config_prompts = {}
