#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 run_endpoint_suite.py --lane baseline --endpoint "${BASELINE_ENDPOINT:-http://localhost:8084}" --model "${BASELINE_MODEL:-qwen36-27b-llamacpp-baseline}" "$@"
