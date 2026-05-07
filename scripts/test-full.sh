#!/bin/bash
# Tier: full — correctness suite with live Kestrel, no third-party or benchmarks (~45s)
# Matches what CI runs on every PR.
set -e
cd "$(dirname "$0")/.."
uv run pytest -v -m "not third_party and not performance"
