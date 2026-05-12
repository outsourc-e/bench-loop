#!/usr/bin/env bash
set -euo pipefail
# Env-configurable because PC1 may have buun-llama-cpp in different locations.
PORT="${PORT:-8084}"
HOST="${HOST:-0.0.0.0}"
MODEL_REPO="${MODEL_REPO:-unsloth/Qwen3.6-35B-A3B-GGUF}"
MODEL_FILE="${MODEL_FILE:-Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
MODEL_ALIAS="${MODEL_ALIAS:-qwen36-llama-cpp-baseline}"
CTX="${CTX:-32768}"
NGPU_LAYERS="${NGPU_LAYERS:--1}"
BATCH="${BATCH:-2048}"
UBATCH="${UBATCH:-512}"
LLAMA_SERVER="${LLAMA_SERVER:-}"

if [[ -z "$LLAMA_SERVER" ]]; then
  for candidate in "$HOME/buun-llama-cpp/build/bin/llama-server" "$HOME/llama.cpp/build/bin/llama-server" "$(command -v llama-server || true)"; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then LLAMA_SERVER="$candidate"; break; fi
  done
fi
if [[ -z "$LLAMA_SERVER" || ! -x "$LLAMA_SERVER" ]]; then
  echo "llama-server not found. Set LLAMA_SERVER=/path/to/buun-llama-cpp/build/bin/llama-server" >&2
  exit 2
fi

exec "$LLAMA_SERVER" \
  --host "$HOST" \
  --port "$PORT" \
  --hf-repo "$MODEL_REPO" \
  --hf-file "$MODEL_FILE" \
  --alias "$MODEL_ALIAS" \
  -c "$CTX" \
  -ngl "$NGPU_LAYERS" \
  -b "$BATCH" \
  -ub "$UBATCH"
