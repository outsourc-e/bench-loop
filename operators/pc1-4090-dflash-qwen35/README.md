# PC1 4090 DFlash Qwen3.5 operator package

Purpose: validate Qwen3.5-27B + z-lab DFlash draft on PC1 / RTX 4090 against a bare Qwen3.5-27B vLLM baseline.

This is intentionally small: launch endpoints, run 5 deterministic prompts x 3 repeats, record raw outputs and speed, then manually decide quality. No dashboards. No ceremony. Science, not vibes.

## Paths

Repo:

/Users/aurora/.openclaw/workspace/bench-loop

Operator package:

/Users/aurora/.openclaw/workspace/bench-loop/operators/pc1-4090-dflash-qwen35

Files:

- recipe.json — lane config and pass/fail gates
- prompts.json — deterministic prompt suite and manual criteria
- run_operator.py — OpenAI-compatible runner for DFlash vs baseline
- run.sh — thin wrapper around run_operator.py
- launch_baseline_8084.sh — bare vLLM baseline launcher
- benchloop-record-template.json — fields to copy into BenchLoop after review

## First commands on PC1

Terminal A — DFlash candidate:

```bash
source ~/venv-dflash/bin/activate
bash ~/dflash-vllm-launch.sh
```

Terminal B — verify DFlash:

```bash
curl http://localhost:8083/v1/models
```

Terminal C — bare baseline:

```bash
cd /Users/aurora/.openclaw/workspace/bench-loop/operators/pc1-4090-dflash-qwen35
chmod +x launch_baseline_8084.sh run.sh
./launch_baseline_8084.sh
```

Terminal D — verify baseline and run suite:

```bash
curl http://localhost:8084/v1/models
cd /Users/aurora/.openclaw/workspace/bench-loop/operators/pc1-4090-dflash-qwen35
./run.sh
```

Smoke-only mode, if baseline is not ready yet:

```bash
cd /Users/aurora/.openclaw/workspace/bench-loop/operators/pc1-4090-dflash-qwen35
./run.sh --skip-baseline --repeats 1
```

## Success signals

DFlash can pass only if all are true:

- Endpoint probe: `http://localhost:8083/v1/models` returns 200 and the expected model id.
- Operator run completes all DFlash requests: 5 prompts x 3 repeats = 15 OK requests.
- Baseline run completes all baseline requests: 15 OK requests.
- DFlash output has zero hard-fail pattern hits.
- Manual quality review passes 5/5 prompts and 3/3 repeats per prompt.
- JSON prompt is parseable JSON only.
- Code prompts are syntactically sane.
- No malformed imports, decorators, braces, route syntax, quote corruption, early cutoff, or nonsense tokens.
- DFlash mean completion tok/s is >= 1.5x baseline.
- No CUDA OOM, vLLM crash, request hang, or server death.

Strong win:

- Quality pass
- Reliability pass
- Speedup >= 2.0x

Marginal/research-only:

- Quality pass
- Reliability pass
- Speedup 1.3x-1.5x

## Failure signals

Stop early and mark failed if any of these appear:

- `from fastapi = FastAPI`
- malformed decorators like `@app.get("/health():`
- broken route strings/braces/quotes
- invalid JSON on JSON-only prompt
- repeated output instability at temperature 0
- completion is fast but visibly corrupted
- DFlash endpoint crashes, hangs, or OOMs
- speedup < 1.3x

Decision rule:

- Fail quality = DFlash lane dead for now, regardless of tok/s.
- Fail speed but pass quality = keep as research, do not productize.
- Fail reliability = only revisit if logs show a trivial config/runtime fix.
- Pass all gates = DFlash lane alive.

## Output

Each run writes:

operators/pc1-4090-dflash-qwen35/runs/<UTC-run-id>/operator-result.json

The file includes:

- raw responses
- latency
- completion token counts
- approximate completion tok/s
- hard-fail hits
- DFlash vs baseline summary
- BenchLoop record fields
- manual review placeholder

## What to record for BenchLoop

Copy/update these fields from `operator-result.json`:

- lane: `pc1-4090-dflash-qwen35`
- hardware.machine: `PC1`
- hardware.gpu: `RTX 4090`
- runtime.engine/version/env: `vLLM` / `0.19.1` / `~/venv-dflash`
- target_model: `Qwen3.5-27B`
- draft_model: `z-lab/Qwen3.5-27B-DFlash`
- endpoint: `http://localhost:8083`
- baseline_endpoint: `http://localhost:8084`
- prompts: `5`
- repeats: `3`
- temperature: `0`
- dflash mean completion tok/s
- baseline mean completion tok/s
- speedup_vs_baseline
- quality_pass: true/false after manual review
- reliability_pass: true/false
- decision: pass / fail_quality / fail_speed / fail_reliability / research_only
- raw_result_path
- notes: include any exact corruption examples or OOM/crash logs
