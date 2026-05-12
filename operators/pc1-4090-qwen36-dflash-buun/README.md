# PC1 4090 Qwen3.6 DFlash / buun-llama-cpp operator package

Purpose: run nonstop, quality-first Qwen3.6 experiments on PC1 and emit BenchLoop-ingestable artifacts.

Primary model path:
- Repo: unsloth/Qwen3.6-35B-A3B-GGUF
- First quant: Qwen3.6-35B-A3B-UD-Q4_K_M.gguf (~22.1GB)
- Fallback quants if VRAM/runtime fails: MXFP4_MOE, UD-IQ4_NL_XL, UD-Q3_K_XL

Artifacts:
- recipe.json: lane config and gates
- prompts.json: deterministic quality/speed smoke suite
- run_operator.py: one-shot or nonstop candidate-vs-baseline runner
- run_once.sh: one cycle
- run_forever.sh: continuous loop, stops on hard quality/reliability failure
- launch_qwen36_buun_8083.sh: env-configurable candidate server launcher
- launch_qwen36_baseline_8084.sh: env-configurable baseline launcher
- pc1_preflight.sh: read-only PC1 readiness check for package, GPU, llama-server, cache hints, and endpoint probes
- runs/<timestamp>/operator-result.json: immutable raw result bundle
- runs/benchloop-ingest.jsonl: append-only records for BenchLoop import

Quality gates are intentionally strict. A speedup with corrupted code/JSON is a failed experiment.

## PC1 operator sequence

Terminal A - candidate server:

```bash
cd /Users/aurora/.openclaw/workspace/bench-loop/operators/pc1-4090-qwen36-dflash-buun
chmod +x *.sh run_operator.py
# Optional if auto-discovery misses the binary:
# export LLAMA_SERVER=$HOME/buun-llama-cpp/build/bin/llama-server
./launch_qwen36_buun_8083.sh
```

Terminal B - baseline server:

```bash
cd /Users/aurora/.openclaw/workspace/bench-loop/operators/pc1-4090-qwen36-dflash-buun
# If candidate and baseline cannot both fit on one 4090, run baseline first, then candidate, or lower baseline ctx/batch.
PORT=8084 MODEL_ALIAS=qwen36-llama-cpp-baseline ./launch_qwen36_baseline_8084.sh
```

Terminal C - smoke:

```bash
cd /Users/aurora/.openclaw/workspace/bench-loop/operators/pc1-4090-qwen36-dflash-buun
./pc1_preflight.sh
curl -fsS http://localhost:8083/v1/models
curl -fsS http://localhost:8084/v1/models
./run_once.sh --repeats 1
```

Terminal C - nonstop loop:

```bash
./run_forever.sh --sleep 60
```

If baseline cannot stay resident with candidate:

```bash
./run_once.sh --skip-baseline --repeats 3
```

Then run baseline in a separate window/time slice and merge by comparing median completion tok/s manually before promotion.

## BenchLoop consumption

Every cycle writes:

```text
runs/<UTC>/operator-result.json
runs/benchloop-ingest.jsonl
```

Use operator-result.json as the source of truth. Only mark a run BenchLoop-ready after manual_review is updated:
- quality_pass=true
- reliability_pass=true
- speed_pass=true
- decision=pass

Then import with:

```bash
cd /Users/aurora/.openclaw/workspace/bench-loop
python3 scripts/ingest_operator_result.py operators/pc1-4090-qwen36-dflash-buun/runs/<UTC>/operator-result.json
```
