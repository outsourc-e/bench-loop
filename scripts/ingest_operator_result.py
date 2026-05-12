#!/usr/bin/env python3
"""Import a PC1 operator-result.json into BenchLoop's local run store.

This is intentionally conservative: pending manual-review runs are rejected unless
--allow-pending is passed. BenchLoop should not publish fast garbage.
"""
from __future__ import annotations
import argparse, json, re, shutil
from datetime import datetime
from pathlib import Path


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-") or "run"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("operator_result", type=Path)
    p.add_argument("--out-root", type=Path, default=Path.home()/".bench-loop"/"runs")
    p.add_argument("--allow-pending", action="store_true")
    args = p.parse_args()
    bundle = json.loads(args.operator_result.read_text(encoding="utf-8"))
    review = bundle.get("manual_review", {})
    if not args.allow_pending:
        if review.get("quality_pass") is not True or review.get("reliability_pass") is not True or review.get("speed_pass") is not True or review.get("decision") != "pass":
            raise SystemExit("Refusing import: manual_review must be quality_pass=true, reliability_pass=true, speed_pass=true, decision=pass. Use --allow-pending for private drafts only.")
    recipe = bundle["recipe"]
    cand = bundle["summary"]["candidate"]
    fields = bundle.get("benchloop_record_fields", {})
    timestamp = bundle.get("created_at") or datetime.now().isoformat()
    model_id = fields.get("candidate", {}).get("model") or recipe.get("candidate", {}).get("model") or recipe["id"]
    run = {
        "version": "operator-import-v1",
        "timestamp": timestamp,
        "model": {"model_id": model_id, "family": recipe.get("model", {}).get("family", "Qwen3.6"), "parameter_count": "35B-A3B", "quantization": recipe.get("baseline", {}).get("file", "")},
        "machine": {"machine_id": recipe.get("machine", "PC1"), "cpu": "", "gpu": recipe.get("hardware", {}).get("gpu", "RTX 4090"), "gpu_memory_gb": recipe.get("hardware", {}).get("vram_gb", 24), "system_memory_gb": 0.0, "os": "", "backend": recipe.get("runtime", {}).get("primary_engine", "buun-llama-cpp")},
        "provider": "openai_compat",
        "harness": "raw/operator",
        "harness_version": recipe["id"],
        "total_runtime_sec": 0.0,
        "overall_score": 0.0,
        "quality_score": 100.0 if review.get("quality_pass") is True else 0.0,
        "speed_score": min((cand.get("median_completion_tokens_per_sec") or 0), 100.0),
        "reliability_score": 100.0 if review.get("reliability_pass") is True else 0.0,
        "value_score": 0.0,
        "speed_metrics": {"ttft_ms": 0.0, "prompt_eval_tok_per_sec": 0.0, "generation_tok_per_sec": cand.get("median_completion_tokens_per_sec") or cand.get("mean_completion_tokens_per_sec") or 0.0, "total_latency_ms": cand.get("mean_latency_ms") or 0.0},
        "suites": {},
        "operator_metadata": {"schema": bundle.get("schema"), "run_id": bundle.get("run_id"), "lane": recipe["id"], "speedup_vs_baseline": cand.get("speedup_vs_baseline"), "auto_quality_clean": review.get("auto_quality_clean"), "source_path": str(args.operator_result.resolve()), "decision": review.get("decision"), "notes": review.get("notes", "")}
    }
    run_dir = args.out_root / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{slug(model_id)}-pc1-operator"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir/"run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    shutil.copy2(args.operator_result, run_dir/"operator-result.json")
    print(run_dir/"run.json")
    return 0

if __name__ == "__main__": raise SystemExit(main())
