#!/usr/bin/env python3
"""
Quick & dirty throughput bench for an OpenAI-compat llama-server endpoint.

Hits /v1/chat/completions with a fixed prompt set, captures timings, computes:
  peak tok/s (over all runs)
  avg tok/s (mean across runs)
  p50/p95 tok/s
  draft acceptance (when reported by /v1/completions? llama.cpp exposes via response `timings`)
  prompt_eval tok/s
  total prompts / errors

Usage:
  python bench_speed.py --url http://100.90.212.55:8081 --tag win8k --repeats 5 \
      --max-tokens 512 --out bench_results.jsonl
"""
from __future__ import annotations
import argparse
import json
import os
import statistics
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime

PROMPTS = [
    ("short_list", "List the 10 most widely spoken human languages by number of native speakers. Just numbered list, no commentary."),
    ("code_fastapi", "Write a FastAPI endpoint /health that returns {status: ok, uptime_seconds: <int>} using process start time. Include imports."),
    ("reverse_fn", "Write a Python function reverse_string(s: str) -> str that reverses the string without using slicing. Include a docstring and one test call."),
    ("transformer_explain", "Explain transformer attention briefly (queries, keys, values, softmax, context vector). 4-6 sentences max."),
    ("capital_quiz", "What is the capital of Australia, and why do people often mistakenly say Sydney? One short paragraph."),
]


def call_once(url: str, prompt: str, max_tokens: int, timeout: int, temperature: float = 0.0, thinking: bool = False, model: str = "qwen"):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if not thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    t1 = time.perf_counter()

    usage = payload.get("usage", {}) or {}
    timings = payload.get("timings", {}) or {}
    completion_tokens = usage.get("completion_tokens", 0)
    prompt_tokens = usage.get("prompt_tokens", 0)
    elapsed = t1 - t0
    # prefer server-side timings if available
    eval_ms = timings.get("predicted_ms")
    prompt_ms = timings.get("prompt_ms")
    eval_tps = timings.get("predicted_per_second")
    prompt_tps = timings.get("prompt_per_second")

    if not eval_tps and completion_tokens and elapsed > 0:
        eval_tps = completion_tokens / elapsed
    return {
        "elapsed_s": round(elapsed, 3),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "eval_ms": eval_ms,
        "prompt_ms": prompt_ms,
        "gen_tps": round(eval_tps or 0, 2),
        "prompt_tps": round(prompt_tps or 0, 2),
        "finish_reason": (payload.get("choices") or [{}])[0].get("finish_reason"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://100.90.212.55:8081")
    ap.add_argument("--tag", required=True, help="label for this config (e.g. win8k-v1)")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--warmup", action="store_true", help="run one warm-up shot per prompt first")
    ap.add_argument("--thinking", action="store_true", help="leave thinking ON")
    ap.add_argument("--out", default="bench_results.jsonl")
    args = ap.parse_args()

    # Warmup to ensure model and cache hot
    if args.warmup:
        print("[warmup] running 1 pass...", flush=True)
        for pid, prompt in PROMPTS:
            try:
                call_once(args.url, prompt, 64, args.timeout, thinking=args.thinking)
            except Exception as e:  # noqa
                print(f"  warmup {pid}: {e}", flush=True)

    runs = []
    start = datetime.utcnow().isoformat()
    print(f"[bench] tag={args.tag} url={args.url} repeats={args.repeats} max_tokens={args.max_tokens}", flush=True)
    for r in range(1, args.repeats + 1):
        for pid, prompt in PROMPTS:
            try:
                res = call_once(args.url, prompt, args.max_tokens, args.timeout, thinking=args.thinking)
                res.update({"tag": args.tag, "run": r, "prompt_id": pid})
                runs.append(res)
                print(f"  run{r} {pid:22s} gen_tps={res['gen_tps']:>7} comp_tok={res['completion_tokens']:>4} elapsed={res['elapsed_s']}s finish={res['finish_reason']}", flush=True)
            except urllib.error.URLError as e:
                print(f"  run{r} {pid}: URLError {e}", flush=True)
            except Exception as e:  # noqa
                print(f"  run{r} {pid}: ERR {type(e).__name__}: {e}", flush=True)

    if not runs:
        print("no successful runs")
        sys.exit(1)

    tps = [r["gen_tps"] for r in runs if r["gen_tps"] > 0]
    prompt_tps = [r["prompt_tps"] for r in runs if r["prompt_tps"] > 0]
    summary = {
        "tag": args.tag,
        "started_utc": start,
        "url": args.url,
        "repeats": args.repeats,
        "max_tokens": args.max_tokens,
        "n_runs": len(runs),
        "gen_tps_peak": max(tps) if tps else None,
        "gen_tps_mean": round(statistics.mean(tps), 2) if tps else None,
        "gen_tps_median": round(statistics.median(tps), 2) if tps else None,
        "gen_tps_stdev": round(statistics.stdev(tps), 2) if len(tps) > 1 else 0,
        "gen_tps_p95": round(statistics.quantiles(tps, n=20)[18], 2) if len(tps) >= 20 else None,
        "prompt_tps_mean": round(statistics.mean(prompt_tps), 2) if prompt_tps else None,
    }
    print("\n=== summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    with open(args.out, "a") as f:
        f.write(json.dumps({"kind": "summary", **summary}) + "\n")
        for r in runs:
            f.write(json.dumps({"kind": "run", **r}) + "\n")
    print(f"\nappended to {args.out}")


if __name__ == "__main__":
    main()
