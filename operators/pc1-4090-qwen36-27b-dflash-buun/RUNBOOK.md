# PC1 remote operator sequence

Workdir:

```bash
cd /Users/aurora/.openclaw/workspace/bench-loop/operators/pc1-4090-qwen36-27b-dflash-buun
chmod +x *.sh *.py
```

## 0. Build/verify buun-llama-cpp

```bash
cd ~
if [ ! -d buun-llama-cpp ]; then
  git clone https://github.com/spiritbuun/buun-llama-cpp.git
fi
cd ~/buun-llama-cpp
git fetch origin master
git checkout master
git rev-parse --short HEAD
# Must include commit b9d01582b or newer. If not, stop.
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON
cmake --build build --config Release -j "$(nproc)"
export LLAMA_SERVER=$HOME/buun-llama-cpp/build/bin/llama-server
$LLAMA_SERVER --version || true
```

## 1. Baseline first, sequential

Terminal A:

```bash
cd /Users/aurora/.openclaw/workspace/bench-loop/operators/pc1-4090-qwen36-27b-dflash-buun
export LLAMA_SERVER=$HOME/buun-llama-cpp/build/bin/llama-server
export MODEL_DIR=$HOME/models/qwen36-27b-dflash-buun
./launch_baseline_8084.sh 2>&1 | tee runs/baseline-server.log
```

Terminal B:

```bash
cd /Users/aurora/.openclaw/workspace/bench-loop/operators/pc1-4090-qwen36-27b-dflash-buun
curl -fsS http://localhost:8084/v1/models
./run_baseline_suite.sh --repeats 3
```

Save the printed `runs/<UTC>-baseline/baseline-result.json` path as:

```bash
export BASELINE_JSON=/Users/aurora/.openclaw/workspace/bench-loop/operators/pc1-4090-qwen36-27b-dflash-buun/runs/<UTC>-baseline/baseline-result.json
```

Stop Terminal A with Ctrl-C.

## 2. Candidate DFlash second

Terminal A:

```bash
cd /Users/aurora/.openclaw/workspace/bench-loop/operators/pc1-4090-qwen36-27b-dflash-buun
export LLAMA_SERVER=$HOME/buun-llama-cpp/build/bin/llama-server
export MODEL_DIR=$HOME/models/qwen36-27b-dflash-buun
./launch_candidate_8083.sh 2>&1 | tee runs/candidate-server.log
```

Terminal B:

```bash
cd /Users/aurora/.openclaw/workspace/bench-loop/operators/pc1-4090-qwen36-27b-dflash-buun
curl -fsS http://localhost:8083/v1/models
./run_candidate_suite.sh --repeats 3
```

Save the printed `runs/<UTC>-candidate/candidate-result.json` path as:

```bash
export CANDIDATE_JSON=/Users/aurora/.openclaw/workspace/bench-loop/operators/pc1-4090-qwen36-27b-dflash-buun/runs/<UTC>-candidate/candidate-result.json
```

## 3. Merge and judge

```bash
cd /Users/aurora/.openclaw/workspace/bench-loop/operators/pc1-4090-qwen36-27b-dflash-buun
./compare_lane_results.py "$BASELINE_JSON" "$CANDIDATE_JSON"
```

This writes:

```text
runs/<UTC>-comparison/operator-result.json
runs/benchloop-ingest.jsonl
```

## Fast failure rules

Kill immediately if any candidate run shows:

```text
HARD_FAIL
AUTO_FAIL on JSON/code/reasoning
from fastapi = FastAPI
@app.get("/health():
replacement characters / <0x...>
repeated junk tokens
server crash / CUDA OOM / request hang
```

Do not keep tuning a corrupt lane.

## Promotion rules

Research continuation:
- auto_quality_clean=true
- candidate median tok/s >= 1.3x baseline

BenchLoop-ready win:
- manual quality_pass=true
- reliability_pass=true
- speedup_vs_baseline >= 1.5x
- ideally 3 consecutive clean comparison bundles

Strong win:
- speedup_vs_baseline >= 2.0x with clean quality

## If first pass fails

- If quality corrupts: kill DFlash/buun lane and fall back to clean no-draft Qwen3.6-27B baseline/KV experiments.
- If quality passes but speedup <1.3x: keep as research-only; next try target `Qwen3.6-27B-UD-Q4_K_XL.gguf` because spiritbuun used UD-Q4_K_XL in its README benchmark.
- If speed passes but reliability fails: inspect server logs; only retry if it is an obvious config/VRAM issue.
