#!/usr/bin/env bash
set -euo pipefail

# PC1 readiness check for the Qwen3.6 / 4090 / buun-DFlash BenchLoop operator lane.
# Safe: read-only checks plus optional endpoint probes. Does not launch servers.

PORT_CANDIDATE="${PORT_CANDIDATE:-8083}"
PORT_BASELINE="${PORT_BASELINE:-8084}"
MODEL_REPO="${MODEL_REPO:-unsloth/Qwen3.6-35B-A3B-GGUF}"
MODEL_FILE="${MODEL_FILE:-Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

status=0

section() { printf '\n== %s ==\n' "$*"; }
pass() { printf 'PASS: %s\n' "$*"; }
warn() { printf 'WARN: %s\n' "$*"; }
fail() { printf 'FAIL: %s\n' "$*"; status=1; }

section "operator package"
for f in recipe.json prompts.json run_operator.py run_once.sh run_forever.sh launch_qwen36_buun_8083.sh launch_qwen36_baseline_8084.sh; do
  [[ -f "$f" ]] && pass "$f present" || fail "$f missing"
done
python3 -m py_compile run_operator.py && pass "run_operator.py compiles" || fail "run_operator.py compile failed"
bash -n *.sh && pass "shell scripts parse" || fail "shell script syntax failed"

section "GPU / CUDA"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader || true
  pass "nvidia-smi available"
else
  fail "nvidia-smi not found; this is probably not PC1 or NVIDIA drivers are unavailable"
fi

section "llama-server discovery"
LLAMA_SERVER="${LLAMA_SERVER:-}"
if [[ -z "$LLAMA_SERVER" ]]; then
  for candidate in "$HOME/buun-llama-cpp/build/bin/llama-server" "$HOME/llama.cpp/build/bin/llama-server" "$(command -v llama-server || true)"; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
      LLAMA_SERVER="$candidate"
      break
    fi
  done
fi
if [[ -n "$LLAMA_SERVER" && -x "$LLAMA_SERVER" ]]; then
  pass "llama-server: $LLAMA_SERVER"
  "$LLAMA_SERVER" --version 2>/dev/null || true
else
  fail "llama-server not found. Set LLAMA_SERVER=$HOME/buun-llama-cpp/build/bin/llama-server or build/install llama.cpp."
fi

section "Hugging Face / model cache hints"
if command -v huggingface-cli >/dev/null 2>&1; then
  pass "huggingface-cli available"
elif command -v hf >/dev/null 2>&1; then
  pass "hf CLI available"
else
  warn "HF CLI not found; llama-server can still download with --hf-repo if built with HF support."
fi
printf 'Expected repo: %s\nExpected file: %s\n' "$MODEL_REPO" "$MODEL_FILE"
for cache in "$HOME/.cache/huggingface/hub" "${HF_HOME:-}/hub"; do
  [[ -n "${cache:-}" && "$cache" != "/hub" && -d "$cache" ]] || continue
  if find "$cache" -name "$MODEL_FILE" -print -quit 2>/dev/null | grep -q .; then
    pass "model file appears cached under $cache"
  else
    warn "model file not found under $cache"
  fi
done

section "ports / endpoint probes"
for port in "$PORT_CANDIDATE" "$PORT_BASELINE"; do
  if command -v lsof >/dev/null 2>&1 && lsof -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    pass "port $port has a listener"
  else
    warn "port $port has no listener yet"
  fi
  if curl -fsS --max-time 2 "http://localhost:$port/v1/models" >/tmp/pc1-preflight-models-$port.json 2>/tmp/pc1-preflight-curl-$port.err; then
    pass "http://localhost:$port/v1/models reachable"
    python3 - <<PY 2>/dev/null || true
import json
p='/tmp/pc1-preflight-models-$port.json'
print(json.dumps(json.load(open(p)).get('data', [])[:3], indent=2)[:1000])
PY
  else
    warn "http://localhost:$port/v1/models not reachable yet"
  fi
done

section "next command"
cat <<'EOF'
Terminal A:
  cd /Users/aurora/.openclaw/workspace/bench-loop/operators/pc1-4090-qwen36-dflash-buun
  chmod +x *.sh run_operator.py
  ./launch_qwen36_buun_8083.sh

Terminal B:
  cd /Users/aurora/.openclaw/workspace/bench-loop/operators/pc1-4090-qwen36-dflash-buun
  PORT=8084 MODEL_ALIAS=qwen36-llama-cpp-baseline ./launch_qwen36_baseline_8084.sh

Terminal C:
  cd /Users/aurora/.openclaw/workspace/bench-loop/operators/pc1-4090-qwen36-dflash-buun
  ./pc1_preflight.sh
  ./run_once.sh --repeats 1
EOF

exit "$status"
