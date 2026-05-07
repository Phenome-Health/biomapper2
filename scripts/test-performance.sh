#!/bin/bash
# Tier: performance — cold-cache per-step benchmarks (~70s)
# The clear_kestrel_cache fixture deletes the HTTP cache before the run
# so timings reflect real network latency. Pass --kestrel-url and --tag
# to benchmark a specific KG build:
#   ./scripts/test-performance.sh --kestrel-url https://staging.example.com/api --tag spoke-merged
set -e
cd "$(dirname "$0")/.."
uv run pytest -m performance -v -s "$@"
