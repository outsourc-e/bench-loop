#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${MODEL_DIR:-$HOME/models/qwen36-27b-dflash-buun}"
TARGET_REPO="${TARGET_REPO:-unsloth/Qwen3.6-27B-GGUF}"
TARGET_FILE="${TARGET_FILE:-Qwen3.6-27B-Q4_K_M.gguf}"
DRAFT_REPO="${DRAFT_REPO:-spiritbuun/Qwen3.6-27B-DFlash-GGUF}"
DRAFT_FILE="${DRAFT_FILE:-dflash-draft-3.6-q8_0.gguf}"
LLAMA_SERVER="${LLAMA_SERVER:-}"
if [[ -z "$LLAMA_SERVER" ]]; then
  for candidate in "$HOME/buun-llama-cpp/build/bin/llama-server" "$HOME/llama.cpp/build/bin/llama-server" "$(command -v llama-server || true)"; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then LLAMA_SERVER="$candidate"; break; fi
  done
fi
if [[ -z "$LLAMA_SERVER" || ! -x "$LLAMA_SERVER" ]]; then
  echo "llama-server not found. Build spiritbuun/buun-llama-cpp or set LLAMA_SERVER." >&2
  exit 2
fi
mkdir -p "$MODEL_DIR/target" "$MODEL_DIR/draft"
if [[ ! -f "$MODEL_DIR/target/$TARGET_FILE" ]]; then
  hf download "$TARGET_REPO" "$TARGET_FILE" --local-dir "$MODEL_DIR/target"
fi
if [[ "${NEED_DRAFT:-0}" == "1" && ! -f "$MODEL_DIR/draft/$DRAFT_FILE" ]]; then
  hf download "$DRAFT_REPO" "$DRAFT_FILE" --local-dir "$MODEL_DIR/draft"
fi
TARGET_PATH="$MODEL_DIR/target/$TARGET_FILE"
DRAFT_PATH="$MODEL_DIR/draft/$DRAFT_FILE"

PORT="${PORT:-8084}"
HOST="${HOST:-0.0.0.0}"
ALIAS="${MODEL_ALIAS:-qwen36-27b-llamacpp-baseline}"
exec "$LLAMA_SERVER"   -m "$TARGET_PATH"   -ngl 99   -np 1 -c 6048   -fa on -b 256 -ub 64   --host "$HOST" --port "$PORT" --alias "$ALIAS" --jinja   --chat-template-kwargs '{"enable_thinking": false}'
