#!/usr/bin/env bash
set -euo pipefail

source ~/venv-dflash/bin/activate
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3.5-27B \
  --host 0.0.0.0 \
  --port 8084 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code
