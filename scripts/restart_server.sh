#!/bin/bash
# Remote helper: kill server, re-launch with params from env, wait for ready.
# Runs on PC1 WSL Ubuntu.
set -u

: "${MODEL:=/mnt/c/Users/13479/.lmstudio/models/lmstudio-community/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf}"
: "${DRAFT:=/mnt/c/Users/13479/.lmstudio/models/unsloth/Qwen3-1.7B/Qwen3-1.7B-Q4_K_M.gguf}"
: "${CTX:=8192}"
: "${CTX_DRAFT:=4096}"
: "${CTK:=q8_0}"
: "${CTV:=q8_0}"
: "${DRAFT_MAX:=12}"
: "${DRAFT_MIN:=3}"
: "${DRAFT_P_MIN:=0.6}"
: "${PORT:=8081}"
: "${EXTRA:=}"
: "${TAG:=run}"

pkill -9 -f llama-server 2>/dev/null
sleep 3

TS=$(date +%s)
LOG=/tmp/llama-${TAG}-${TS}.log
echo "LOG=$LOG" > /tmp/llama-last-log

# Wait for VRAM to free
for i in 1 2 3 4 5 6; do
  MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "${MEM:-99999}" -lt 1500 ] && break
  sleep 2
done

# Compose -md only if DRAFT set
DRAFT_ARGS=""
if [ -n "${DRAFT:-}" ] && [ "$DRAFT" != "none" ]; then
  DRAFT_ARGS="-md $DRAFT -ngld 99 -cd $CTX_DRAFT --draft-max $DRAFT_MAX --draft-min $DRAFT_MIN --draft-p-min $DRAFT_P_MIN"
fi

nohup ~/ik_llama.cpp/build/bin/llama-server \
  -m "$MODEL" \
  $DRAFT_ARGS \
  -ngl 99 -c $CTX \
  --jinja \
  -fa on -ctk $CTK -ctv $CTV \
  --host 0.0.0.0 --port $PORT \
  $EXTRA \
  > "$LOG" 2>&1 &
disown
PID=$!
echo "PID=$PID"
echo "LOG=$LOG"

for i in $(seq 1 120); do
  if curl -sf --max-time 2 "http://127.0.0.1:${PORT}/v1/models" > /dev/null 2>&1; then
    echo "ready after ${i}s"
    exit 0
  fi
  if ! kill -0 $PID 2>/dev/null; then
    echo "PROCESS DIED after ${i}s"
    tail -40 "$LOG"
    exit 2
  fi
  sleep 1
done
echo "TIMEOUT 120s"
tail -30 "$LOG"
exit 1
