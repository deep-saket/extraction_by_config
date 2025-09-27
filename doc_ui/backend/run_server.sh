#!/usr/bin/env bash
set -euo pipefail
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")"/../.. && pwd)}"
python -m uvicorn doc_ui.backend.server:app --host 0.0.0.0 --port 8001 --reload

