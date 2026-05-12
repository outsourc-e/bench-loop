# BenchLoop

<p align="center">
  <img src="https://raw.githubusercontent.com/outsourc-e/bench-loop-web/main/site/public/og-image.png" alt="BenchLoop" width="640" />
</p>

<p align="center">
  <a href="https://bench-loop.com"><img src="https://img.shields.io/badge/site-bench-loop.com-2dd47f?style=flat-square" alt="site" /></a>
  <a href="https://github.com/outsourc-e/bench-loop/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-2dd47f?style=flat-square" alt="MIT" /></a>
  <img src="https://img.shields.io/badge/status-beta-eab308?style=flat-square" alt="beta" />
</p>

**Benchmark local LLMs by what actually matters.**

BenchLoop is a local-first CLI + web app for benchmarking LLMs running on your own hardware. It scores models across seven repeatable suites — quality, speed, reliability, agentic tool use, coding, instruction following — and gives you receipts: per-task outputs, latency, token counts, machine info, scores.

No accounts, no telemetry, no API keys. Your model, your machine, your numbers.

```
$ benchloop run --model qwen3:8b --suites speed,toolcall,agent
... 8 tasks, 4 tools, 6 turns avg, 74.6 tok/s ...

Overall  73.4  ████████░░
Quality  73.6  ████████░░
Speed    78.9  █████████░
Agent    96.9  █████████▌
```

Published runs live at <https://bench-loop.com/leaderboard>.

## Why

Hosted LLM leaderboards answer *"which model wins on a server farm someone else paid for?"* BenchLoop answers *"which model + harness + hardware combination actually works for me right now?"* — the question you have when picking a local stack.

It is repeatable on purpose: every run persists to disk, the task set is frozen, the scorer is deterministic. If you say "qwen3:8b scored 89 on my 4090", anyone can install BenchLoop and verify it.

## Install

### pipx (recommended)

```bash
pipx install benchloop
benchloop --version
```

### pip

```bash
pip install benchloop
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
Overall = 0.55 · quality + 0.20 · speed + 0.25 · reliability
```

- **Quality** = mean of non-speed suite scores (size-fair).
- **Speed** = `12.54 · log2(tok/s) + 0.9`, clamped to 0–100.
- **Reliability** = pass rate across all tasks.
- **Agent** = `correct_final + efficient + no_hallucinated_tools + all_required_called`, 25 pts each, averaged across tasks.

## Local web app

A FastAPI backend + React frontend bundle ships alongside the CLI for visualizing runs:

```bash
benchloop dashboard   # starts the local web app on :5180
```

Tabs: Models, Benchmark, Leaderboard, Compare runs, Chat, agent trace viewer.

## Publish a run

Run locally, export to the public leaderboard JSON, open a PR:

```bash
benchloop export --output my-runs.json
# then PR against outsourc-e/bench-loop with the JSON
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

BenchLoop is **v0.1 beta**. The benchmark surface, scoring, web app, agent loop, and four harnesses all work end-to-end. Stuff still on the roadmap:

- Streaming TTFT for OpenAI-compatible providers (currently 0 on those backends — ollama TTFT is fine)
- Bigger task fixtures (each suite is intentionally small and frozen for v1)
- Hosted submission flow for community runs
- More provider adapters (TGI, Bedrock, etc. if there's demand)

## License

MIT. See `LICENSE`.
