#!/usr/bin/env bash
# Launch the Streamlit demo. Creates a venv and installs deps on first run.
set -euo pipefail

# Resolve script dir robustly, whether invoked as `./run.sh`, `bash run.sh`, or `. run.sh`.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd -- "$SCRIPT_DIR"

if [ ! -x ".venv/bin/python" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    ./.venv/bin/python -m pip install --upgrade pip
fi

# Self-heal: install deps if streamlit is missing (e.g. a prior partial install).
if ! ./.venv/bin/python -c "import streamlit" >/dev/null 2>&1; then
    echo "Installing dependencies..."
    ./.venv/bin/python -m pip install -r requirements.txt
fi

exec ./.venv/bin/python -m streamlit run app.py
