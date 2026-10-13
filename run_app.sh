#!/bin/bash
# Risk Desk — one-click launcher
# Open a terminal here and type: bash run_app.sh

set -e
PROJ="$(cd "$(dirname "$0")" && pwd)"
VENV="$PROJ/.venv"
PYTHON="$VENV/bin/python"

echo ""
echo "╔══════════════════════════════════╗"
echo "║   Risk Desk — AI Risk Analysis   ║"
echo "╚══════════════════════════════════╝"
echo ""
echo "Project: $PROJ"

# Verify .venv python works
if ! "$PYTHON" --version &>/dev/null; then
    echo "ERROR: Cannot find .venv Python. Is uv installed?"
    echo "Try: uv python install 3.11 && uv sync"
    exit 1
fi

# Install Flask if missing (uses uv since venv has no pip)
if ! "$PYTHON" -c "import flask" 2>/dev/null; then
    echo "Installing Flask into .venv …"
    if command -v uv &>/dev/null; then
        uv pip install flask==3.1.3 gunicorn==26.2.0 --python "$PYTHON" -q
    else
        "$PYTHON" -m pip install flask==3.1.3 gunicorn==26.2.0 -q
    fi
    echo "Flask installed ✓"
fi

# Open browser after a moment
(sleep 2 && { xdg-open http://127.0.0.1:5000 || open http://127.0.0.1:5000; } 2>/dev/null) &

echo "✓ Starting at http://127.0.0.1:5000"
echo "  Press Ctrl+C to stop."
echo ""
cd "$PROJ/app"
"$PYTHON" app.py
