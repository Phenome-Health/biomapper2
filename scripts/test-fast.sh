#!/bin/bash
# Tier: fast — unit tests only, no API calls (~18s)
set -e
cd "$(dirname "$0")/.."
uv run pytest -v -m "not requires_api and not third_party and not performance"
