#!/usr/bin/env python3
"""Continuous PC1 operator for Qwen3.6 buun/DFlash experiments.

Runs candidate vs baseline OpenAI-compatible endpoints, writes one immutable
operator-result.json per cycle, and appends a BenchLoop ingest JSONL record.
"""
from __future__ import annotations
import argparse, ast, json, re, statistics, sys, time, urllib.error, urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_RECIPE = ROOT / "recipe.json"
DEFAULT_PROMPTS = ROOT / "prompts.json"
DEFAULT_OUT = ROOT / "runs"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def request_json(method: str, url: str, payload: dict[str, Any] | None, timeout: int) -> tuple[int, dict[str, Any], str]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type":"application/json"}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return resp.status, json.loads(raw), raw
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try: body = json.loads(raw)
        except Exception: body = {"error": raw}
        return exc.code, body, raw
    except Exception as exc:
        return 0, {"error": str(exc)}, ""


def probe(endpoint: str, timeout: int) -> dict[str, Any]:
    status, body, raw = request_json("GET", endpoint.rstrip("/") + "/v1/models", None, timeout)
    models = [x.get("id") for x in body.get("data", []) if isinstance(x, dict) and x.get("id")] if isinstance(body, dict) else []
    return {"status": status, "ok": 200 <= status < 300, "models": models, "raw": raw[:4000]}


def chat(endpoint: str, model: str, prompt: str, temperature: float, max_tokens: int, timeout: int) -> dict[str, Any]:
    payload = {"model": model, "temperature": temperature, "max_tokens": max_tokens, "messages": [{"role":"user", "content": prompt}]}
    started = time.perf_counter()
    status, body, raw = request_json("POST", endpoint.rstrip("/") + "/v1/chat/completions", payload, timeout)
    elapsed = time.perf_counter() - started
    choice = (body.get("choices") or [{}])[0] if isinstance(body, dict) else {}
    msg = choice.get("message") or {}
    usage = body.get("usage") or {} if isinstance(body, dict) else {}
    content = msg.get("content") or choice.get("text") or ""
    ctok = int(usage.get("completion_tokens") or 0)
    ptok = int(usage.get("prompt_tokens") or 0)
    return {"status": status, "ok": 200 <= status < 300 and bool(content.strip()), "error": "" if 200 <= status < 300 else str(body.get("error", body))[:1000], "content": content, "raw_response": body, "raw_response_text": raw[:20000], "latency_ms": round(elapsed*1000,2), "prompt_tokens": ptok, "completion_tokens": ctok, "completion_tokens_per_sec": round(ctok/elapsed,2) if elapsed > 0 and ctok else 0.0}


def hard_fail_hits(content: str, patterns: list[str]) -> list[str]:
    hits = [p for p in patterns if p and p in content]
    regexes = [r"from\s+\w+\s*=\s*\w+", r"@app\.get\([^\n]*:\s*$", r"<0x[0-9A-Fa-f]+>", r"(.)\1{24,}"]
    for pat in regexes:
        if re.search(pat, content, flags=re.MULTILINE): hits.append("regex:" + pat)
    return sorted(set(hits))


def auto_checks(prompt_id: str, content: str) -> dict[str, Any]:
    c = content.strip()
    out: dict[str, Any] = {"auto_pass": None, "checks": []}
    try:
        if prompt_id in {"json_object_only", "csv_to_json", "tool_json_call"}:
            parsed = json.loads(c)
            out["checks"].append("json_parse_ok")
            if prompt_id == "json_object_only": out["auto_pass"] = isinstance(parsed, dict) and isinstance(parsed.get("name"), str) and isinstance(parsed.get("age"), (int,float)) and isinstance(parsed.get("tags"), list)
            elif prompt_id == "csv_to_json": out["auto_pass"] = isinstance(parsed, list) and len(parsed) == 2
            elif prompt_id == "tool_json_call": out["auto_pass"] = isinstance(parsed, dict) and parsed.get("tool") == "send_email" and parsed.get("arguments", {}).get("to") == "sam@example.com"
        elif prompt_id in {"fastapi_health", "fib_with_tests"}:
            ast.parse(c)
            out["checks"].append("python_ast_ok")
            out["auto_pass"] = "```" not in c
        elif prompt_id == "small_reasoning":
            out["auto_pass"] = c == "67"
        elif prompt_id == "exact_five_bullets":
            bullets = [line for line in c.splitlines() if re.match(r"^\s*(-|\*|\d+[.)])\s+", line)]
            out["auto_pass"] = len(bullets) == 5
        elif prompt_id == "specdec_tokenizer_120_words":
            words = re.findall(r"\b\w+\b", c)
            out["auto_pass"] = 90 <= len(words) <= 150 and "token" in c.lower()
    except Exception as exc:
        out["checks"].append(f"auto_check_error:{exc}")
        out["auto_pass"] = False
    return out


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in results if r.get("ok")]
    tps = [r["completion_tokens_per_sec"] for r in ok if r.get("completion_tokens_per_sec",0) > 0]
    lat = [r["latency_ms"] for r in ok]
    hard = sum(1 for r in results if r.get("hard_fail_hits"))
    auto_known = [r for r in results if r.get("auto", {}).get("auto_pass") is not None]
    auto_pass = sum(1 for r in auto_known if r.get("auto", {}).get("auto_pass") is True)
    return {"requests": len(results), "ok_requests": len(ok), "error_count": len(results)-len(ok), "hard_fail_count": hard, "auto_checked": len(auto_known), "auto_pass_count": auto_pass, "auto_pass_rate": round(auto_pass/len(auto_known),3) if auto_known else None, "mean_completion_tokens_per_sec": round(sum(tps)/len(tps),2) if tps else 0.0, "median_completion_tokens_per_sec": round(statistics.median(tps),2) if tps else 0.0, "mean_latency_ms": round(sum(lat)/len(lat),2) if lat else 0.0, "total_completion_tokens": sum(int(r.get("completion_tokens") or 0) for r in results)}


def run_cycle(args: argparse.Namespace, cycle_index: int | None = None) -> tuple[Path, dict[str, Any]]:
    recipe, prompts = load_json(args.recipe), load_json(args.prompts)
    shape = recipe["run_shape"]
    cand_ep = args.candidate_endpoint or recipe["candidate"]["endpoint"]
    cand_model = args.candidate_model or recipe["candidate"]["model"]
    base_ep = args.baseline_endpoint or recipe["baseline"]["endpoint"]
    base_model = args.baseline_model or recipe["baseline"]["model"]
    repeats = args.repeats if args.repeats is not None else int(shape["repeats"])
    max_tokens = args.max_tokens if args.max_tokens is not None else int(shape["max_tokens"])
    temp = args.temperature if args.temperature is not None else float(shape["temperature"])
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run {run_id} cycle={cycle_index if cycle_index is not None else '-'}")
    endpoint_probe = {"candidate": probe(cand_ep, args.timeout)}
    if not args.skip_baseline: endpoint_probe["baseline"] = probe(base_ep, args.timeout)
    for lane,p in endpoint_probe.items(): print(f"  probe {lane}: status={p['status']} ok={p['ok']} models={p['models'][:3]}")
    if not endpoint_probe["candidate"]["ok"]: raise SystemExit("candidate endpoint unreachable")
    if not args.skip_baseline and not endpoint_probe["baseline"]["ok"]: raise SystemExit("baseline endpoint unreachable")
    lanes = [("candidate", cand_ep, cand_model)] + ([] if args.skip_baseline else [("baseline", base_ep, base_model)])
    all_results = []
    for lane, endpoint, model in lanes:
        print(f"Lane: {lane}")
        for prompt in prompts:
            for repeat in range(1, repeats+1):
                r = chat(endpoint, model, prompt["prompt"], temp, max_tokens, args.timeout)
                r.update({"lane": lane, "endpoint": endpoint, "model": model, "prompt_id": prompt["id"], "prompt_category": prompt.get("category",""), "repeat": repeat, "manual_pass_criteria": prompt.get("manual_pass_criteria", []), "hard_fail_hits": hard_fail_hits(r["content"], prompt.get("hard_fail_patterns", [])), "auto": auto_checks(prompt["id"], r["content"])})
                all_results.append(r)
                sig = "ERR" if not r["ok"] else ("HARD_FAIL" if r["hard_fail_hits"] else ("AUTO_FAIL" if r["auto"].get("auto_pass") is False else "OK"))
                print(f"  {prompt['id']} #{repeat}: {sig} {r['completion_tokens_per_sec']} tok/s {r['latency_ms']} ms")
    by_lane = {lane: [r for r in all_results if r["lane"] == lane] for lane,_,_ in lanes}
    summary = {lane: summarize(items) for lane, items in by_lane.items()}
    if "baseline" in summary and summary["baseline"]["median_completion_tokens_per_sec"] > 0:
        summary["candidate"]["speedup_vs_baseline"] = round(summary["candidate"]["median_completion_tokens_per_sec"] / summary["baseline"]["median_completion_tokens_per_sec"], 3)
    else:
        summary["candidate"]["speedup_vs_baseline"] = None
    auto_quality_clean = summary["candidate"]["error_count"] == 0 and summary["candidate"]["hard_fail_count"] == 0 and summary["candidate"].get("auto_pass_rate") == 1.0
    speedup = summary["candidate"].get("speedup_vs_baseline")
    bundle = {"schema":"benchloop.operator_result.v2", "run_id":run_id, "cycle_index": cycle_index, "created_at":datetime.now(timezone.utc).isoformat(), "recipe":recipe, "endpoint_probe":endpoint_probe, "summary":summary, "results":all_results, "manual_review":{"quality_pass": None, "auto_quality_clean": auto_quality_clean, "reliability_pass": summary["candidate"]["error_count"] == 0, "speed_pass": None if speedup is None else speedup >= 1.5, "decision":"pending_manual_quality_review", "notes":"Manual review remains required before BenchLoop-ready promotion."}, "benchloop_record_fields":{"lane":recipe["id"], "hardware":recipe["hardware"], "runtime":recipe["runtime"], "model":recipe["model"], "candidate":recipe["candidate"], "baseline":recipe["baseline"], "prompts":len(prompts), "repeats":repeats, "temperature":temp, "quality_pass":None, "auto_quality_clean":auto_quality_clean, "reliability_pass":summary["candidate"]["error_count"] == 0, "speedup":speedup, "decision":"pending", "raw_result_path": str(out_dir / "operator-result.json")}}
    result_path = out_dir / "operator-result.json"
    result_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    ingest = args.out_dir / "benchloop-ingest.jsonl"
    with ingest.open("a", encoding="utf-8") as f: f.write(json.dumps(bundle["benchloop_record_fields"]) + "\n")
    print(f"Saved: {result_path}")
    print(f"Ingest JSONL: {ingest}")
    print("Candidate summary:", json.dumps(summary["candidate"], indent=2))
    return result_path, bundle


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE); p.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS); p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--candidate-endpoint"); p.add_argument("--candidate-model"); p.add_argument("--baseline-endpoint"); p.add_argument("--baseline-model")
    p.add_argument("--repeats", type=int); p.add_argument("--max-tokens", type=int); p.add_argument("--temperature", type=float); p.add_argument("--timeout", type=int, default=240)
    p.add_argument("--skip-baseline", action="store_true"); p.add_argument("--forever", action="store_true"); p.add_argument("--sleep", type=int, default=None); p.add_argument("--stop-on-fail", action="store_true")
    args = p.parse_args()
    recipe = load_json(args.recipe); sleep_s = args.sleep if args.sleep is not None else int(recipe["run_shape"].get("sleep_between_cycles_sec", 60))
    cycle = 1
    while True:
        try:
            _, bundle = run_cycle(args, cycle if args.forever else None)
            cand = bundle["summary"]["candidate"]
            failed = cand["error_count"] or cand["hard_fail_count"] or cand.get("auto_pass_rate") not in (None, 1.0)
            if args.stop_on_fail and failed: return 3
        except KeyboardInterrupt:
            print("Interrupted."); return 130
        except SystemExit as exc:
            print(f"FAIL: {exc}")
            if args.stop_on_fail or not args.forever: return 2
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            if args.stop_on_fail or not args.forever: return 1
        if not args.forever: return 0
        cycle += 1
        print(f"Sleeping {sleep_s}s before next cycle...")
        time.sleep(sleep_s)

if __name__ == "__main__": sys.exit(main())
