# PC1 4090 Qwen3.6-27B DFlash / buun-llama-cpp operator package

Mission: validate the highest-upside current lane: `spiritbuun/buun-llama-cpp` + `spiritbuun/Qwen3.6-27B-DFlash-GGUF` against the clean Qwen3.6-27B no-draft baseline on PC1 / RTX 4090.

This is quality-first. If output corrupts, kill the lane immediately. Fast garbage is just garbage with a GPU budget.

## Exact assets

Runtime:
- Fork: `https://github.com/spiritbuun/buun-llama-cpp`
- Branch: `master`
- Minimum commit: `b9d01582b` (SWA support for DFlash drafter; older builds can load and still produce garbage)
- Build flags: `-DGGML_CUDA=ON -DGGML_NATIVE=ON -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON`

Target model:
- Repo: `unsloth/Qwen3.6-27B-GGUF`
- First target file: `Qwen3.6-27B-Q4_K_M.gguf` (~16.8GB)
- Alt target if we want to mirror spiritbuun's README benchmark: `Qwen3.6-27B-UD-Q4_K_XL.gguf` (~17.6GB)

DFlash drafter:
- Repo: `spiritbuun/Qwen3.6-27B-DFlash-GGUF`
- First draft file: `dflash-draft-3.6-q8_0.gguf` (~1.85GB)
- Avoid first: `dflash-draft-3.6-q4_k_m.gguf`; upstream card says acceptance drops from ~43% to ~28%.

Important runtime footgun:
- Disable Qwen thinking: `--chat-template-kwargs '{"enable_thinking": false}'`
- Upstream says thinking-on collapses DFlash acceptance.

## Files in this package

- `recipe.json` — exact lane config and gates
- `prompts.json` — strict deterministic prompt suite
- `launch_baseline_8084.sh` — no-draft baseline server
- `launch_candidate_8083.sh` — DFlash candidate server
- `run_endpoint_suite.py` — run the prompt suite against one endpoint and write lane artifacts
- `compare_lane_results.py` — merge baseline + candidate artifacts into one operator result
- `run_baseline_suite.sh` / `run_candidate_suite.sh` — convenience wrappers
- `runs/<UTC>/...` — immutable artifacts

## First comparison shape

Run sequentially, not concurrently. A single 24GB 4090 cannot hold two full Qwen3.6-27B targets plus DFlash draft comfortably.

1. Start baseline server on `8084`.
2. Run the 8-prompt x 3-repeat suite against baseline.
3. Stop baseline.
4. Start candidate DFlash server on `8083`.
5. Run the same suite against candidate.
6. Merge artifacts and judge speed only if quality is clean.

## Success criteria

Pass only if all are true:
- candidate has 0 request errors
- candidate has 0 hard-fail pattern hits
- candidate auto-pass rate is 1.0
- manual review agrees output is sane
- no CUDA OOM/crash/hang
- median candidate completion tok/s >= 1.5x median baseline completion tok/s

Continue research but do not productize if speedup is 1.3x-1.5x.
Kill immediately if any code/JSON corruption appears.
