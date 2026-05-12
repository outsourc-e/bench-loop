# BenchLoop v0.1 Quick Release Plan

## Goal

Ship a credible beta release that people can install, run locally, and understand:

- CLI install works
- Local web app works
- Public site at bench-loop.com explains the product
- Public leaderboard has real seed data
- Benchmark scores are useful: speed, quality, reliability, agent loop

## Current release artifacts

- Core package: `dist/benchloop-0.1.0-py3-none-any.whl`
- Source package: `dist/benchloop-0.1.0.tar.gz`
- Public site: `../bench-loop-web/site/dist/`
- Local web app: `../bench-loop-web/ui/dist/`
- Public leaderboard JSON: `../bench-loop-web/site/public/data/leaderboard.json`

## Repos

Recommended public repos:

- `outsourc-e/bench-loop` — Python CLI + benchmark suites/scorers/providers/harnesses
- `outsourc-e/bench-loop-web` — FastAPI local API, React local dashboard, static public site

## Pre-launch checklist

### Core CLI

- [x] `benchloop --version`
- [x] `benchloop info`
- [x] `benchloop suites`
- [x] `benchloop export`
- [x] package builds with hatchling
- [x] wheel installs into a fresh venv
- [x] tasks included in wheel
- [x] MIT license
- [x] README explains install, run, suites, harnesses, scoring

### Benchmark quality

- [x] speed suite returns generation tok/s on Ollama
- [x] openai_compat computes tok/s from token usage + wall clock
- [x] agent suite performs real multi-turn tool execution
- [x] harnesses: raw / hermes / qwen / pi
- [x] public JSON includes full, quality-only, and agent-only runs
- [ ] streaming TTFT for openai_compat (post-beta)
- [ ] bigger fixture set (post-beta)

### Local web app

- [x] Models tab
- [x] Benchmark tab with suite selection + harness picker
- [x] Leaderboard tab
- [x] Compare tab
- [x] Run detail page
- [x] Agent trace viewer (turn-by-turn tools + results)
- [x] per-endpoint serial queue in API
- [x] useful traceback on failures
- [x] production build passes

### Public site

- [x] Landing page
- [x] Docs page
- [x] Download page
- [x] Public leaderboard page
- [x] Cloudflare Pages config
- [x] Vercel fallback config
- [x] OpenGraph SVG asset placeholder
- [ ] final pixel-chip logo assets from Eric
- [ ] convert/generated OG PNG once logo is done
- [ ] deploy to Cloudflare/Vercel
- [ ] connect `bench-loop.com` + `www.bench-loop.com`

## Recommended launch order

1. Push `bench-loop` to GitHub.
2. Push `bench-loop-web` to GitHub.
3. Deploy `bench-loop-web/site` to Cloudflare Pages.
4. Connect `bench-loop.com` and `www.bench-loop.com`.
5. Publish `benchloop` to PyPI when ready.
6. Tweet with the seed results:
   - pc1-coder-v2: agent 96.9, overall 74.3, 111 tok/s
   - qwen3:8b: full 72.9, agent 93.8
   - MiniMax M2.7 Small JANGTQ: agent 96.9, full 50.6 on Studio

## Commands

### Build core package

```bash
cd bench-loop
python -m build --wheel --sdist
```

### Smoke test package

```bash
python3 -m venv /tmp/benchloop-test
/tmp/benchloop-test/bin/pip install dist/benchloop-0.1.0-py3-none-any.whl
/tmp/benchloop-test/bin/benchloop info
```

### Export public leaderboard

```bash
cd bench-loop-web/site
node scripts/export-leaderboard.mjs
npm run build
```

### Deploy public site to Cloudflare Pages

```bash
cd bench-loop-web/site
npm run build
npx wrangler pages deploy dist --project-name=benchloop
```

Then add custom domains in Cloudflare Pages:

- `bench-loop.com`
- `www.bench-loop.com`

## Known beta caveats

- OpenAI-compatible TTFT is not streamed yet, so TTFT is `0` for MLX/Osaurus/vLLM-style endpoints.
- Some Ollama models do not support tool calling and will fail the `agent` suite cleanly.
- Full benchmark is intentionally small. Useful for quick local comparison, not a definitive academic eval.
- Concurrent benchmark requests are serialized per endpoint; this is intentional to avoid local server saturation.
