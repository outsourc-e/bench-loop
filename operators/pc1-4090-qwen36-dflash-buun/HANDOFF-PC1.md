# PC1 handoff: 4090 / Qwen3.6 / DFlash-buun / BenchLoop

State: staged locally; no PC1 benchmark result has been produced yet.

## Staged assets

Package root:

```text
/Users/aurora/.openclaw/workspace/bench-loop/operators/pc1-4090-qwen36-dflash-buun
```

Files:

```text
README.md
HANDOFF-PC1.md
recipe.json
prompts.json
run_operator.py
run_once.sh
run_forever.sh
launch_qwen36_buun_8083.sh
launch_qwen36_baseline_8084.sh
pc1_preflight.sh
```

BenchLoop importer:

```text
/Users/aurora/.openclaw/workspace/bench-loop/scripts/ingest_operator_result.py
```

## Experiment tuple

```text
Machine: PC1
GPU: RTX 4090, 24GB
Candidate endpoint: http://localhost:8083
Candidate model alias: qwen36-buun-dflash
Candidate engine: buun-llama-cpp / DFlash-style path, via llama-server-compatible OpenAI API
Baseline endpoint: http://localhost:8084
Baseline model alias: qwen36-llama-cpp-baseline
Baseline engine: plain llama.cpp-style server
Model repo: unsloth/Qwen3.6-35B-A3B-GGUF
First model file: Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
Fallback files: MXFP4_MOE, UD-IQ4_NL_XL, UD-Q3_K_XL
Run shape: 8 prompts x 3 repeats, temperature=0, max_tokens=768
```

## Immediate command sequence

Terminal A: candidate server

```bash
cd /Users/aurora/.openclaw/workspace/bench-loop/operators/pc1-4090-qwen36-dflash-buun
chmod +x *.sh run_operator.py

# If autodiscovery misses it:
# export LLAMA_SERVER=$HOME/buun-llama-cpp/build/bin/llama-server

./pc1_preflight.sh
./launch_qwen36_buun_8083.sh
```

Terminal B: baseline server

```bash
cd /Users/aurora/.openclaw/workspace/bench-loop/operators/pc1-4090-qwen36-dflash-buun
PORT=8084 MODEL_ALIAS=qwen36-llama-cpp-baseline ./launch_qwen36_baseline_8084.sh
```

Terminal C: smoke / run

```bash
cd /Users/aurora/.openclaw/workspace/bench-loop/operators/pc1-4090-qwen36-dflash-buun
./pc1_preflight.sh
curl -fsS http://localhost:8083/v1/models
curl -fsS http://localhost:8084/v1/models
./run_once.sh --repeats 1
```

If smoke is clean:

```bash
./run_once.sh --repeats 3
```

If that is clean for one cycle and there are no hard fails:

```bash
./run_forever.sh --sleep 60
```

If candidate + baseline cannot co-reside in 24GB VRAM:

```bash
./run_once.sh --skip-baseline --repeats 3
```

Then run the baseline in a separate time slice and compare candidate median completion tok/s against baseline median completion tok/s manually before promotion.

## Quality kill switch

Immediately stop treating the lane as a win if any candidate output contains:

```text
from fastapi = FastAPI
malformed FastAPI decorators/routes
invalid JSON for JSON-only prompts
markdown fences where forbidden
replacement characters
repeated nonsense tokens
early cutoff
prompt-fidelity failure
```

Raw tok/s is irrelevant until quality passes.

## Promotion gate

BenchLoop-ready only if:

```text
candidate endpoint reachable
baseline endpoint reachable or intentionally skipped for sequential run
candidate error_count == 0
candidate hard_fail_count == 0
auto_pass_rate == 1.0 for auto-checked prompts
manual quality review passes all prompts/repeats
reliability_pass == true
speedup_vs_baseline >= 1.5
ideally 3 consecutive clean cycles
```

Thresholds:

```text
>=2.0x baseline: strong win
>=1.5x baseline: BenchLoop-ready speed win if quality/reliability pass
1.3x-1.5x: research continuation only
<1.3x: fail
any quality failure: fail regardless of speed
```

## Artifacts to collect

Runner writes:

```text
runs/<UTC>/operator-result.json
runs/benchloop-ingest.jsonl
```

Source of truth is always `operator-result.json`.

After manual review, edit the winning `operator-result.json` only if it actually passed:

```json
"manual_review": {
  "quality_pass": true,
  "reliability_pass": true,
  "speed_pass": true,
  "decision": "pass",
  "notes": "Manual review passed all prompt outputs; no corruption observed."
}
```

Then import:

```bash
cd /Users/aurora/.openclaw/workspace/bench-loop
python3 scripts/ingest_operator_result.py operators/pc1-4090-qwen36-dflash-buun/runs/<UTC>/operator-result.json
```

For private draft import only:

```bash
python3 scripts/ingest_operator_result.py operators/pc1-4090-qwen36-dflash-buun/runs/<UTC>/operator-result.json --allow-pending
```

## Current blocker

Needs PC1 execution. Local package validates syntactically, but this machine does not prove CUDA fit, llama-server path, buun behavior, or real output quality.
