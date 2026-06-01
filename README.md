# BenchLoop

<p align="center">
  <img src="https://raw.githubusercontent.com/outsourc-e/bench-loop-web/main/site/public/og-image.png" alt="BenchLoop" width="640" />
</p>

<p align="center">
  <a href="https://bench-loop.com"><img src="https://img.shields.io/badge/site-bench--loop.com-2dd47f?style=flat-square" alt="site" /></a>
  <a href="https://pypi.org/project/benchloop-cli/"><img src="https://img.shields.io/pypi/v/benchloop-cli?style=flat-square&color=2dd47f" alt="pypi" /></a>  <a href="https://github.com/outsourc-e/bench-loop/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-2dd47f?style=flat-square" alt="MIT" /></a>
  <img src="https://img.shields.io/badge/status-beta-eab308?style=flat-square" alt="beta" />
</p>

**Benchmark local LLMs by what actually matters.**

BenchLoop is a local-first CLI + web app for benchmarking LLMs running on your own hardware or cloud providers. It scores models across seven repeatable suites — quality, speed, reliability, agentic tool use, coding, instruction following — and gives you receipts: per-task outputs, latency, token counts, machine info, scores.

No accounts, no telemetry. Local models need no API keys; cloud providers use standard OpenAI-compatible auth. Your model, your machine (or your provider), your numbers.

```
$ benchloop run --model qwen3:8b --suites speed,toolcall,agent
... 8 tasks, 4 tools, 6 turns avg, 74.6 tok/s ...

Overall  73.4  ████████░░
Quality  73.6  ████████░░
Speed    78.9  █████████░
Agent    96.9  █████████▌
```

Published runs live at <https://bench-loop.com/leaderboard>. Every completed local benchmark auto-publishes there.
## Why

Hosted LLM leaderboards answer *"which model wins on a server farm someone else paid for?"* BenchLoop answers *"which model + harness + hardware combination actually works for me right now?"* — the question you have when picking a local stack.

It is repeatable on purpose: every run persists to disk, the task set is frozen, the scorer is deterministic. If you say "qwen3:8b scored 89 on my 4090", anyone can install BenchLoop and verify it.

## Install

### pipx (recommended)

```bash
pipx install benchloop-cli
benchloop --version
```

> The PyPI distribution is named `benchloop-cli` (the bare `benchloop` name was taken by an unrelated dataset library). The installed commands are still `benchloop` and `bench-loop`.

### pip

```bash
pip install benchloop-cli
```

### From source

```bash
git clone https://github.com/outsourc-e/bench-loop
cd bench-loop
pip install -e .
```

## Run your first benchmark

Make sure you have a local LLM endpoint running. Anything OpenAI-compatible or Ollama-flavored works:

- Ollama at `http://localhost:11434` (default)
- LM Studio at `http://localhost:1234` (`--provider openai_compat`)
- MLX / Osaurus at `http://localhost:8000` (`--provider openai_compat`)
- vLLM, Jan, llama-server, etc.

Then:

```bash
benchloop run \
  --model qwen3:8b \
  --endpoint http://localhost:11434 \
  --provider ollama
```

This runs every default suite, scores them, prints a console report, and persists the full run to `~/.bench-loop/runs/`.

### Run a subset

```bash
benchloop run --model qwen3:8b --suites speed,agent
```

### Different prompting harness

Same model, four ways to talk to it:

```bash
benchloop run --model qwen3:8b --harness raw      # native tool calling
benchloop run --model qwen3:8b --harness hermes   # <tool_call>{...}</tool_call>
benchloop run --model qwen3:8b --harness qwen     # <function_call>{...}</function_call>
benchloop run --model qwen3:8b --harness pi       # <think>...</think> + Hermes tags
```

### Stamp custom hardware (e.g. when benchmarking through a tunnel)

```bash
benchloop run \
  --model qwen3:8b \
  --endpoint http://localhost:11435 \
  --hardware "NVIDIA RTX 4090 24GB" \
  --gpu "NVIDIA RTX 4090" \
  --gpu-memory-gb 24
```

### Benchmark cloud/remote APIs

Works with any OpenAI-compatible endpoint — DashScope, OpenRouter, Together, OpenAI, vLLM with auth, sglang, etc.

```bash
# Via environment variable
export OPENAI_API_KEY="sk-..."
benchloop run \
  --model qwen3.7-max \
  --provider openai_compat \
  --endpoint https://dashscope-intl.aliyuncs.com/compatible-mode \
  --remote

# Or inline
benchloop run \
  --model gpt-4o \
  --provider openai_compat \
  --endpoint https://api.openai.com/v1 \
  --api-key sk-... \
  --remote
```

The `--remote` flag (auto-detected for non-localhost endpoints) switches to cloud-aware scoring:
- **Speed** uses streaming TTFT (time-to-first-token) + effective content tok/s
- **Overall** = 0.50·quality + 0.25·speed + 0.25·reliability (vs local's 0.55/0.20/0.25)
- Reasoning models: content tok/s excludes internal thinking tokens

### API key auth

Required for vLLM, sglang, and most cloud providers. Two ways to provide it:

```bash
# 1. Environment variable (recommended)
export OPENAI_API_KEY="your-key-here"
benchloop run --model your-model --provider openai_compat --endpoint http://your-server:8000

# 2. CLI flag
benchloop run --model your-model --provider openai_compat --endpoint http://your-server:8000 --api-key your-key-here
```

The CLI flag takes precedence over the env var. For Ollama and local providers without auth, neither is needed.

When you use multiple OpenAI-compatible endpoints, set per-endpoint keys with
`BENCHLOOP_OPENAI_KEYS`. Entries are comma-separated `endpoint=key` pairs and
the endpoint must match the base URL without a trailing slash:

```bash
export BENCHLOOP_OPENAI_KEYS="http://127.0.0.1:8000=sk-local,https://openrouter.ai/api=sk-or-..."
```

BenchLoop uses the matching endpoint-specific key first, then falls back to
`OPENAI_API_KEY`.

### Launch the local dashboard

v0.2.0+ ships the full FastAPI + React dashboard inside the wheel. After `pipx install benchloop-cli`:

```bash
benchloop dashboard
# → open http://127.0.0.1:8877
```

Need it to survive browser/terminal churn? Print a service template instead of keeping the dashboard tied to one shell:

```bash
benchloop dashboard --service-template launchd
benchloop dashboard --service-template systemd
benchloop dashboard --service-template windows-task
```

This serves the Models, Benchmark, Leaderboard, Compare, and Chat tabs on a single port, with auto-discovered local providers (Ollama, LM Studio, MLX/Osaurus, vLLM, Jan).

For hot-reload development against a clone of [`bench-loop-web`](https://github.com/outsourc-e/bench-loop-web):

```bash
benchloop dashboard --dev
```

## Suites

| Suite | What it scores |
|---|---|
| `speed` | Latency, throughput, TTFT, generation tok/s across short/medium/long contexts |
| `toolcall` | Structured tool-call correctness across realistic tasks (weather, stocks, email, search) |
| `coding` | Executable Python tasks verified in a sandboxed subprocess (10s timeout) |
| `dataextract` | JSON / structured extraction from messy natural language |
| `instructfollow` | Constraint following, formatting, exactness |
| `reasonmath` | Small reasoning + math tasks with deterministic checks |
| `agent` | **Multi-turn agentic tool use.** BenchLoop drives a real loop: model emits a tool call, BenchLoop executes it locally, feeds the result back, model iterates until done. Scores correctness, efficiency, no-hallucination, required-tool coverage. |

## Scoring

```
Local:  Overall = 0.55 · quality + 0.20 · speed + 0.25 · reliability
Cloud:  Overall = 0.50 · quality + 0.25 · speed + 0.25 · reliability  (with streaming speed data)
        Overall = 0.65 · quality + 0.35 · reliability                   (no speed data)
```

- **Quality** = mean of non-speed suite scores (size-fair).
- **Speed (local)** = `12.54 · log2(tok/s) + 0.9`, clamped to 0–100.
- **Speed (cloud)** = 0.60 · TTFT_score + 0.40 · tok/s_score, where TTFT uses exponential decay (200ms→100, 2000ms→40) and tok/s uses a log curve calibrated for 20-150 tok/s.
- **Reliability** = pass rate across all tasks.
- **Agent** = `correct_final + efficient + no_hallucinated_tools + all_required_called`, 25 pts each, averaged across tasks.

## Local web app

A FastAPI backend + React frontend bundle ships alongside the CLI for visualizing runs:

```bash
benchloop dashboard   # starts the local web app on :5180
```

Tabs: Models, Benchmark, Leaderboard, Compare runs, Chat, agent trace viewer.

## Publish a run

Every completed benchmark auto-publishes to <https://bench-loop.com/leaderboard> via `https://api.bench-loop.com/submit`. Runs are deduped by `(machine_id, run_id)` so the same run from the same machine won't be double-counted.

Opt out:

```bash
export BENCHLOOP_NO_SUBMIT=1
```

You can still manually export a snapshot for sharing / archiving:

```bash
benchloop export --output my-runs.json
```

## Architecture

```
bench-loop/                    ← this repo, the CLI + suites + scorers
  bench_loop/
    cli.py                     ← `benchloop` entrypoint
    suites/                    ← speed, toolcall, coding, agent, ...
    harness.py                 ← raw / hermes / qwen / pi adapters
    providers/                 ← ollama, openai_compat
    runner/orchestrator.py     ← drives suites + harnesses
    tasks/                     ← frozen task YAML fixtures
bench-loop-web/                ← the web app (separate repo)
  api/                         ← FastAPI wrapper around bench_loop
  ui/                          ← local dashboard
  site/                        ← public bench-loop.com static site
```

## Status

BenchLoop is **v0.2 beta**. The benchmark surface, scoring, web app, agent loop, four harnesses, and cloud provider support all work end-to-end. Stuff still on the roadmap:

- ~~Streaming TTFT for OpenAI-compatible providers~~ ✅ (v0.2.3+ with `--remote`)
- Bigger task fixtures (each suite is intentionally small and frozen for v1)
- Hosted submission flow for community runs
- Cloud-specific leaderboard on bench-loop.com (filter by local vs remote)
- More provider adapters (TGI, Bedrock, etc. if there's demand)

## License

MIT. See `LICENSE`.
