#!/usr/bin/env bash
# Visual Buffet Setup Script
# Run: ./setup.sh

set -e

echo "=== Visual Buffet Setup ==="
echo

# Check for uv, install if missing
if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo
fi

# Navigate to project directory
cd "$(dirname "$0")"

# Remove stale venv if it exists with wrong architecture
if [ -d ".venv" ] && ! .venv/bin/python3 --version &> /dev/null 2>&1; then
    echo "Removing stale virtual environment..."
    rm -rf .venv
fi

# Sync dependencies
echo "Installing dependencies..."
uv sync --extra dev --extra gui

echo
echo "=== Setup Complete ==="
echo
echo "Usage:"
echo "  uv run visual-buffet --help     # Show commands"
echo "  uv run visual-buffet hardware   # Detect hardware"
echo "  uv run visual-buffet gui        # Launch web GUI"
echo "  uv run visual-buffet tag IMAGE  # Tag an image"
echo
echo "To install ML plugins (combine extras in one command):"
echo "  uv sync --extra dev --extra gui --extra ram_plus --extra florence_2 --extra siglip"
echo
echo "Or install one at a time (each replaces previous extras):"
echo "  uv sync --extra ram_plus"
echo "  uv sync --extra florence_2"
echo "  uv sync --extra siglip"
echo
