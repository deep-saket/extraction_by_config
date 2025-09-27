import os
from pathlib import Path

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[3]))
CONFIG_PATH = str(ROOT / "config/files/settings.yml")

