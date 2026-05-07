#!/bin/bash
# Tier: all — performance benchmarks first (cold cache), then full suite including third-party (~120s+)
# Running performance first means the HTTP cache is warmed for subsequent correctness tests.
set -e
cd "$(dirname "$0")/.."

echo "=== Step 1/2: Performance benchmarks (cold cache) ==="
uv run pytest -m performance -v -s

echo ""
echo "=== Step 2/2: Full suite (warm cache) ==="
uv run pytest -v -m "not performance"
