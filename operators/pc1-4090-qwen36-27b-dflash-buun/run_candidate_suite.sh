#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 run_endpoint_suite.py --lane candidate --endpoint "${CANDIDATE_ENDPOINT:-http://localhost:8083}" --model "${CANDIDATE_MODEL:-qwen36-27b-dflash-buun}" "$@"
