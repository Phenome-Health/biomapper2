#!/bin/bash
set -e  # Exit on first error

# Change to project root directory
cd "$(dirname "$0")/.."

echo "Running ruff fixes..."
uv run ruff check --fix

echo "Running black formatting..."
uv run black .

echo "✅ Auto-fixes applied!"
echo ""
echo "Note: Any potential Pyright errors would need manual fixing."
echo "Run ./scripts/check.sh to verify all checks pass."