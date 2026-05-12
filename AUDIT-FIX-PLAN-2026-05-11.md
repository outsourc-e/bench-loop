# BenchLoop Stabilization Plan — 2026-05-11

## Goal
Make BenchLoop solid enough to:
1. connect to local or remote OpenAI-compatible / Ollama-style endpoints,
2. run benchmark suites reliably,
3. persist results locally,
4. show a trustworthy local leaderboard with speed + hardware + quant + flags + benchmark scores,
5. be ready to ship a submission pipeline next.

---

## Product flow we are targeting

1. User installs / opens BenchLoop
2. User either:
   - pulls a model, or
   - connects an existing model endpoint via v1 completions / Ollama
3. User runs a benchmark against selected suites
4. Result is saved locally in canonical run format
5. Leaderboard reads local runs and shows:
   - model id
   - provider / harness
   - quantization
   - hardware
   - speed metrics (ttft, tok/s)
   - runtime flags / endpoint metadata where available
   - benchmark aggregate scores and suite breakdowns
6. Later: optional submit to hosted leaderboard

---

## Current status

## Working enough to build on
- BenchLoop app UI exists
- BenchLoop-specific API routes exist under `/api/benchloop/*`
- Local runs can be loaded from disk
- Local leaderboard rows can be derived from runs
- Model listing works on the corrected BenchLoop backend
- Existing historical runs are visible

## Broken / drifted
- CLI repo missing `bench_loop.harness`
- CLI/report path broken by `HardwareSnapshot` import mismatch
- Default running backend ports were stale / wrong service
- Frontend and backend drifted from docs
- Two overlapping route families exist: `/api/benchmark/*` and `/api/benchloop/*`
- Current leaderboard row shape is too thin for final product surface

---

## P0 — Must fix now

### 1. Canonical runtime contract
Standardize development/runtime on:
- Frontend: `127.0.0.1:5176`
- BenchLoop API: `127.0.0.1:8878`

Required changes:
- add documented boot commands
- optionally add one script to start both reliably
- stop treating 8876/8877/5174 as canonical for BenchLoop

### 2. Repair core CLI compatibility
Required changes:
- restore a real `bench_loop.harness` module
- fix `bench_loop.report.console` vs `hardware.py` interface drift
- get these commands healthy again:
  - `bench-loop info`
  - `bench-loop suites`
  - `bench-loop run ...`

Acceptance criteria:
- CLI works in repo venv without import errors
- backend can call core BenchLoop logic without shims that hide deeper breakage

### 3. Lock one API surface
Canonical API surface should be `/api/benchloop/*` for BenchLoop-specific features.

Required changes:
- audit overlap with `/api/benchmark/*`
- either:
  - keep benchmark routes as legacy wrappers, or
  - migrate UI fully and clearly separate old routes
- update README to match reality

Acceptance criteria:
- app uses one predictable route family
- no ambiguity about which endpoints power BenchLoop

### 4. End-to-end benchmark validation
Run at least two real validations:
- local small model
- PC1 remote Qwen 3.6 27B endpoint

Suggested target:
- PC1 existing Qwen 3.6 27B endpoint via Ollama/OpenAI-compatible API

Acceptance criteria:
- model listed
- benchmark starts
- suites complete
- run saved locally
- leaderboard updates

---

## P1 — Needed before public push

### 5. Enrich leaderboard row schema
Leaderboard needs more than overall score.

Add / verify fields:
- model id
- provider
- harness
- quantization
- machine summary
- GPU / RAM / backend
- runtimeSec
- ttft_ms
- prompt tok/s
- generation tok/s
- total latency
- overall / quality / speed / reliability / value
- suite counts / pass-fail summary
- endpoint / flags where safe

Acceptance criteria:
- leaderboard answers “what model-stack is best on this hardware?” at a glance

### 6. Provider connection UX
Support the intended user story cleanly:
- connect existing endpoint
- verify compatibility
- browse models
- hand selected model into benchmark flow

Acceptance criteria:
- user can point BenchLoop at PC1 and immediately benchmark `qwen3.6-27b`

### 7. Run schema hardening
Ensure local run format is canonical and future-proof.

Needed:
- stable schema version
- machine metadata normalization
- quant/provider/harness consistency
- optional endpoint metadata sanitization

---

## P2 — Next after local product is solid

### 8. Submission pipeline
- submission schema
- local preview
- POST endpoint
- validation and dedupe

### 9. Hosted leaderboard readiness
- ingest local run rows
- merge best-per-stack logic
- spam prevention / auth later

---

## Recommended execution order

### Phase A — Stabilize foundation
1. fix harness module
2. fix CLI/report drift
3. confirm CLI commands work
4. normalize backend/frontend startup
5. update docs

### Phase B — Prove product loop
6. run benchmark on local small model
7. run benchmark on PC1 Qwen 3.6 27B
8. inspect saved run shape
9. enrich leaderboard rendering if needed

### Phase C — Prep for push
10. submission schema/preview cleanup
11. public-facing docs
12. hosted leaderboard follow-up

---

## Immediate next task
Repair the core CLI/runtime drift first.

Reason:
If the repo core is unstable, every app/backend fix above it is fragile.

---

## Known validation target
PC1 already has Qwen 3.6 27B available and should be used as the first meaningful remote benchmark target once the flow is repaired.
