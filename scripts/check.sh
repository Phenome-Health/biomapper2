#!/bin/bash
set -e  # Exit on first error

# Change to project root directory
cd "$(dirname "$0")/.."

echo "Running ruff check..."
uv run ruff check

echo "Running black check..."
uv run black --check .

echo "Running pyright..."
uv run pyright

echo "Running tests..."
uv run pytest -v

echo "✅ All checks passed!"