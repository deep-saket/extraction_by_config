#!/usr/bin/env bash
#
# setup_env.sh
#
# Usage:
#   source /path/to/extraction_by_config/setup_env.sh

# 1. Detect the directory where this script lives
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 2. Export PROJECT_ROOT so subprocesses (like Python) can access it
export PROJECT_ROOT

# 3. Add to PYTHONPATH if not already present
if [[ ":$PYTHONPATH:" != *":$PROJECT_ROOT:"* ]]; then
  export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
fi

echo "✅ PROJECT_ROOT set to: $PROJECT_ROOT"
echo "✅ PYTHONPATH includes: $PYTHONPATH"