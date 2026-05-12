#!/usr/bin/env python3
"""Operator runner for PC1 RTX 4090 Qwen3.5 DFlash validation.

Runs a deterministic prompt suite against two OpenAI-compatible endpoints:
- DFlash candidate on localhost:8083
- bare vLLM baseline on localhost:8084

It records raw outputs, latency, token counts, approximate tokens/sec, hard-fail pattern hits,
and an aggregate BenchLoop-friendly result bundle. Manual quality review is still required;
fast garbage is the bug this package is designed to catch.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_RECIPE = ROOT / "recipe.json"
DEFAULT_PROMPTS = ROOT / "prompts.json"
DEFAULT_OUT = ROOT / "runs"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def post_json(url: str, payload: dict[str, Any], timeout: int) -> tuple[int, dict[str, Any], str]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return resp.status, json.loads(raw), raw
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(raw)
        except Exception:
            body = {"error": raw}
        return exc.code, body, raw
    except Exception as exc:
        return 0, {"error": str(exc)}, ""


def get_json(url: str, timeout: int) -> tuple[int, dict[str, Any], str]:
    req = urllib.request.Request(url, headers={"Content-Type": "application/json"}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return resp.status, json.loads(raw), raw
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(raw)
        except Exception:
            body = {"error": raw}
        return exc.code, body, raw
    except Exception as exc:
        return 0, {"error": str(exc)}, ""


def chat_once(endpoint: str, model: str, prompt: str, temperature: float, max_tokens: int, timeout: int) -> dict[str, Any]:
    url = endpoint.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    started = time.perf_counter()
    status, body, raw = post_json(url, payload, timeout=timeout)
    elapsed = time.perf_counter() - started
    choice = (body.get("choices") or [{}])[0] if isinstance(body, dict) else {}
    message = choice.get("message") or {}
    usage = body.get("usage") or {} if isinstance(body, dict) else {}
    content = message.get("content") or ""
    completion_tokens = int(usage.get("completion_tokens") or 0)
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    approx_tps = completion_tokens / elapsed if elapsed > 0 and completion_tokens else 0.0
    return {
        "status": status,
        "ok": 200 <= status < 300 and bool(content.strip()),
        "error": "" if 200 <= status < 300 else str(body.get("error", body))[:1000],
        "content": content,
        "raw_response": body,
        "raw_response_text": raw[:20000],
        "latency_ms": round(elapsed * 1000, 2),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "completion_tokens_per_sec": round(approx_tps, 2),
    }


def hard_fail_hits(content: str, patterns: list[str]) -> list[str]:
    hits: list[str] = []
    for pattern in patterns:
        if pattern and pattern in content:
            hits.append(pattern)
    malformed_patterns = [
        r"from\s+\w+\s*=\s*\w+",
        r"@app\.get\([^\n]*:\s*$",
        r"```",
    ]
    for pattern in malformed_patterns:
        if re.search(pattern, content, flags=re.MULTILINE):
            hits.append(f"regex:{pattern}")
    return sorted(set(hits))


def probe_endpoint(endpoint: str, timeout: int) -> dict[str, Any]:
    status, body, raw = get_json(endpoint.rstrip("/") + "/v1/models", timeout=timeout)
    models = []
    if isinstance(body, dict):
        models = [item.get("id") for item in body.get("data", []) if isinstance(item, dict) and item.get("id")]
    return {"status": status, "ok": 200 <= status < 300, "models": models, "raw": raw[:5000]}


def summarize_lane(results: list[dict[str, Any]]) -> dict[str, Any]:
    tps_values = [r["completion_tokens_per_sec"] for r in results if r.get("ok") and r.get("completion_tokens_per_sec", 0) > 0]
    latency_values = [r["latency_ms"] for r in results if r.get("ok")]
    error_count = sum(1 for r in results if not r.get("ok"))
    hard_fail_count = sum(1 for r in results if r.get("hard_fail_hits"))
    return {
        "requests": len(results),
        "ok_requests": len(results) - error_count,
        "error_count": error_count,
        "hard_fail_count": hard_fail_count,
        "mean_completion_tokens_per_sec": round(sum(tps_values) / len(tps_values), 2) if tps_values else 0.0,
        "mean_latency_ms": round(sum(latency_values) / len(latency_values), 2) if latency_values else 0.0,
        "total_completion_tokens": sum(int(r.get("completion_tokens") or 0) for r in results),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PC1 4090 DFlash vs baseline operator suite")
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dflash-endpoint", default=None)
    parser.add_argument("--dflash-model", default=None)
    parser.add_argument("--baseline-endpoint", default=None)
    parser.add_argument("--baseline-model", default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--skip-baseline", action="store_true", help="Only run DFlash smoke/suite")
    args = parser.parse_args()

    recipe = load_json(args.recipe)
    prompts = load_json(args.prompts)
    run_shape = recipe["run_shape"]
    dflash_endpoint = args.dflash_endpoint or recipe["dflash"]["endpoint"]
    dflash_model = args.dflash_model or recipe["dflash"]["model"]
    baseline_endpoint = args.baseline_endpoint or recipe["baseline"]["endpoint"]
    baseline_model = args.baseline_model or recipe["baseline"]["model"]
    repeats = args.repeats if args.repeats is not None else int(run_shape["repeats"])
    max_tokens = args.max_tokens if args.max_tokens is not None else int(run_shape["max_tokens"])
    temperature = args.temperature if args.temperature is not None else float(run_shape["temperature"])

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run: {run_id}")
    print(f"DFlash:   {dflash_endpoint} model={dflash_model}")
    if not args.skip_baseline:
        print(f"Baseline: {baseline_endpoint} model={baseline_model}")
    print("Probing endpoints...")

    endpoint_probe = {"dflash": probe_endpoint(dflash_endpoint, args.timeout)}
    if not args.skip_baseline:
        endpoint_probe["baseline"] = probe_endpoint(baseline_endpoint, args.timeout)
    for lane, probe in endpoint_probe.items():
        print(f"  {lane}: status={probe['status']} ok={probe['ok']} models={probe['models'][:5]}")

    if not endpoint_probe["dflash"]["ok"]:
        print("FAIL: DFlash endpoint is not reachable. Start ~/dflash-vllm-launch.sh first.")
        return 2
    if not args.skip_baseline and not endpoint_probe["baseline"]["ok"]:
        print("FAIL: baseline endpoint is not reachable. Start bare vLLM on port 8084 first, or use --skip-baseline for smoke only.")
        return 2

    lanes = [("dflash", dflash_endpoint, dflash_model)]
    if not args.skip_baseline:
        lanes.append(("baseline", baseline_endpoint, baseline_model))

    all_results: list[dict[str, Any]] = []
    for lane, endpoint, model in lanes:
        print(f"\nLane: {lane}")
        for prompt in prompts:
            for repeat in range(1, repeats + 1):
                result = chat_once(endpoint, model, prompt["prompt"], temperature, max_tokens, args.timeout)
                result.update({
                    "lane": lane,
                    "endpoint": endpoint,
                    "model": model,
                    "prompt_id": prompt["id"],
                    "prompt_category": prompt.get("category", ""),
                    "repeat": repeat,
                    "manual_pass_criteria": prompt.get("manual_pass_criteria", []),
                    "hard_fail_hits": hard_fail_hits(result["content"], prompt.get("hard_fail_patterns", [])),
                })
                all_results.append(result)
                signal = "OK"
                if not result["ok"]:
                    signal = "ERR"
                elif result["hard_fail_hits"]:
                    signal = "HARD_FAIL"
                print(
                    f"  {prompt['id']} #{repeat}: {signal} "
                    f"{result['completion_tokens_per_sec']} tok/s "
                    f"{result['latency_ms']} ms "
                    f"tokens={result['completion_tokens']}"
                )

    by_lane = {lane: [r for r in all_results if r["lane"] == lane] for lane, _, _ in lanes}
    summary = {lane: summarize_lane(items) for lane, items in by_lane.items()}
    if "baseline" in summary and summary["baseline"]["mean_completion_tokens_per_sec"] > 0:
        summary["dflash"]["speedup_vs_baseline"] = round(
            summary["dflash"]["mean_completion_tokens_per_sec"] / summary["baseline"]["mean_completion_tokens_per_sec"], 3
        )
    else:
        summary["dflash"]["speedup_vs_baseline"] = None

    bundle = {
        "schema": "benchloop.operator_result.v1",
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "recipe": recipe,
        "endpoint_probe": endpoint_probe,
        "summary": summary,
        "results": all_results,
        "manual_review": {
            "quality_pass": None,
            "reliability_pass": summary["dflash"]["error_count"] == 0,
            "speed_pass": None if summary["dflash"]["speedup_vs_baseline"] is None else summary["dflash"]["speedup_vs_baseline"] >= 1.5,
            "decision": "pending_manual_quality_review",
            "notes": "Set quality_pass only after inspecting outputs for every prompt/repeat. Do not count fast corrupted text as a pass.",
        },
        "benchloop_record_fields": {
            "lane": recipe["id"],
            "hardware": recipe["hardware"],
            "runtime": recipe["runtime"],
            "target_model": recipe["dflash"]["target_model"],
            "draft_model": recipe["dflash"]["draft_model"],
            "endpoint": dflash_endpoint,
            "baseline_endpoint": None if args.skip_baseline else baseline_endpoint,
            "prompts": len(prompts),
            "repeats": repeats,
            "temperature": temperature,
            "quality_pass": None,
            "speedup": summary["dflash"]["speedup_vs_baseline"],
            "reliability_pass": summary["dflash"]["error_count"] == 0,
            "decision": "pending",
        },
    }

    result_path = out_dir / "operator-result.json"
    result_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\nSaved: {result_path}")
    print(f"DFlash summary: {json.dumps(summary['dflash'], indent=2)}")
    if "baseline" in summary:
        print(f"Baseline summary: {json.dumps(summary['baseline'], indent=2)}")
    print("Next: inspect operator-result.json outputs and set manual_review.quality_pass/decision.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
