#!/usr/bin/env bash
#
# setup_env.sh
#
# When sourced, this script will:
#   1) figure out its own directory (i.e. your project root),
#   2) prepend that directory to PYTHONPATH (if not already there).
#
# Usage:
#   source /path/to/extraction_by_config/setup_env.sh
# or you can add this line to your ~/.bashrc / ~/.zshrc:
#   source /Users/your_username/Projects/extraction_by_config/setup_env.sh

# 1. Detect the directory where this script lives
#    (regardless of what directory you're in when you call `source`).
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 2. If PROJECT_ROOT is not already on PYTHONPATH, prepend it.
if [[ ":$PYTHONPATH:" != *":$PROJECT_ROOT:"* ]]; then
  export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
fi

echo "✅ Set PYTHONPATH to include: $PROJECT_ROOT"