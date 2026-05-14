"""Benchmark run management — kick off, stream, list, detail."""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path as FsPath
from typing import Any

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from bench_loop.config import RunConfig
from bench_loop.runner.orchestrator import run_benchmark
from bench_loop.runner.result_writer import save_run
from bench_loop.models import BenchmarkRun

router = APIRouter()

RUNS_DIR = FsPath("~/.bench-loop/runs").expanduser()

# In-memory state for active runs
_active_runs: dict[str, dict[str, Any]] = {}

# Per-endpoint serial queue: concurrent benchmark requests against the same
# endpoint (e.g. one Ollama server) used to race and ReadTimeout. We now
# enqueue them with a per-endpoint asyncio.Lock so one run finishes before the
# next one starts. Different endpoints (e.g. PC1 Ollama + Studio MLX) still
# run in parallel.
_endpoint_locks: dict[str, asyncio.Lock] = {}
_endpoint_queues: dict[str, list[str]] = {}


def _get_endpoint_lock(endpoint: str) -> asyncio.Lock:
    if endpoint not in _endpoint_locks:
        _endpoint_locks[endpoint] = asyncio.Lock()
    return _endpoint_locks[endpoint]


class BenchmarkRequest(BaseModel):
    model: str
    endpoint: str = "http://localhost:11434"
    provider: str = "ollama"
    suites: list[str] = Field(default_factory=lambda: ["speed", "toolcall", "coding", "dataextract", "instructfollow", "reasonmath"])
    harness: str = "raw"
    runs: int = 3
    timeout_sec: float = 300.0
    profile_name: str | None = None
    profile_avatar_url: str | None = None
    profile_url: str | None = None


@router.post("/benchmark/run")
async def start_benchmark_route(req: BenchmarkRequest):
    run_id = str(uuid.uuid4())[:8]
    # Use a plain namespace to stay compatible with multiple RunConfig schemas
    # (the canonical bench_loop uses `base_url`/`suites`/`trials`; the legacy one
    # used `endpoint`/`suite_names`/`runs`). The orchestrator handles both.
    from types import SimpleNamespace
    config = SimpleNamespace(
        model=req.model,
        provider=req.provider,
        endpoint=req.endpoint,
        base_url=req.endpoint,
        harness=req.harness,
        suite_names=req.suites,
        suites=req.suites,
        runs=req.runs,
        trials=req.runs,
        timeout_sec=req.timeout_sec,
    )

    queue = _endpoint_queues.setdefault(req.endpoint, [])
    queue.append(run_id)
    position = max(0, len(queue) - 1)

    _active_runs[run_id] = {
        "status": "queued" if position > 0 else "running",
        "queue_position": position,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "model": req.model,
            "endpoint": req.endpoint,
            "suites": req.suites,
            "harness": req.harness,
        },
        "events": [],
        "result": None,
        "error": None,
    }

    def on_progress(event: dict[str, Any]) -> None:
        _active_runs[run_id]["events"].append(event)

    async def _run():
        lock = _get_endpoint_lock(req.endpoint)
        async with lock:
            # Promote to running once we own the endpoint lock.
            _active_runs[run_id]["status"] = "running"
            _active_runs[run_id]["queue_position"] = 0
            try:
                _endpoint_queues[req.endpoint].remove(run_id)
            except ValueError:
                pass
            await _execute_run(run_id, req, config, on_progress)

    task = asyncio.ensure_future(_run())
    _active_runs[run_id]["task"] = task
    return {"run_id": run_id, "status": _active_runs[run_id]["status"], "queue_position": position}


@router.post("/benchmark/cancel/{run_id}")
async def cancel_benchmark(run_id: str):
    """Cancel an in-flight benchmark run."""
    state = _active_runs.get(run_id)
    if not state:
        return {"ok": False, "error": "run not found"}
    if state.get("status") in ("completed", "failed", "cancelled"):
        return {"ok": True, "already": state["status"]}
    task: asyncio.Task | None = state.get("task")
    if task and not task.done():
        task.cancel()
    state["status"] = "cancelled"
    state["completed_at"] = datetime.now(timezone.utc).isoformat()
    state["error"] = "Cancelled by user"
    state["events"].append({"type": "cancelled", "reason": "user requested"})
    # Drop from queue if still queued
    queue = _endpoint_queues.get(state.get("config", {}).get("endpoint", ""), [])
    if run_id in queue:
        queue.remove(run_id)
    return {"ok": True, "status": "cancelled"}


async def _execute_run(run_id: str, req: "BenchmarkRequest", config: Any, on_progress: Any) -> None:
    try:
        result = await run_benchmark(config, on_progress=on_progress)
        _active_runs[run_id]["status"] = "completed"
        _active_runs[run_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
        _active_runs[run_id]["result"] = result.to_dict()
        try:
            publish_profile = {
                "name": req.profile_name,
                "avatar_url": req.profile_avatar_url,
                "profile_url": req.profile_url,
            }
            saved_path = save_run(result, endpoint=req.endpoint, publish_profile=publish_profile)
            _active_runs[run_id]["saved_path"] = str(saved_path)
        except Exception as save_exc:
            _active_runs[run_id]["events"].append({
                "type": "persist_failed",
                "error": str(save_exc),
            })
    except asyncio.CancelledError:
        # User cancelled — already marked in cancel_benchmark, but make sure.
        if _active_runs[run_id]["status"] not in ("completed", "failed"):
            _active_runs[run_id]["status"] = "cancelled"
        raise
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        err_msg = str(exc) or f"{type(exc).__name__}: {repr(exc)}"
        _active_runs[run_id]["status"] = "failed"
        _active_runs[run_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
        _active_runs[run_id]["error"] = err_msg
        _active_runs[run_id]["traceback"] = tb
        _active_runs[run_id]["events"].append({
            "type": "run_failed",
            "error": err_msg,
            "exception_class": type(exc).__name__,
        })
        print(f"[bench-loop-api] run {run_id} failed:\n{tb}", flush=True)


@router.get("/benchmark/stream/{run_id}")
async def stream_benchmark(run_id: str):
    """SSE stream for benchmark progress — emits granular task-level events."""
    async def event_generator():
        if run_id not in _active_runs:
            yield f"data: {json.dumps({'type': 'error', 'error': 'Run not found'})}\n\n"
            return

        last_idx = 0
        while True:
            run_state = _active_runs.get(run_id)
            if not run_state:
                break

            events = run_state["events"]
            while last_idx < len(events):
                yield f"data: {json.dumps(events[last_idx])}\n\n"
                last_idx += 1

            if run_state["status"] in ("completed", "failed", "cancelled"):
                yield f"data: {json.dumps({'type': 'done', 'status': run_state['status']})}\n\n"
                break

            await asyncio.sleep(0.3)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/benchmark/runs")
async def list_runs(limit: int = Query(default=50, le=200)):
    """List past benchmark runs from disk."""
    if not RUNS_DIR.exists():
        return {"runs": []}

    # "Full benchmark" = at least these quality suites + speed. Coding is bonus.
    REQUIRED_FULL_SUITES = {"speed", "toolcall", "dataextract", "instructfollow", "reasonmath"}

    import math
    def _recompute_speed_score(tok_per_sec: float) -> float:
        """Match bench_loop.suites.speed v2 curve so older runs use the new scale."""
        if tok_per_sec <= 0:
            return 0.0
        return min(100.0, max(0.0, 12.54 * math.log2(tok_per_sec) + 0.9))

    runs = []
    dirs = sorted(RUNS_DIR.iterdir(), reverse=True)
    for d in dirs[:limit]:
        run_file = d / "run.json"
        if not run_file.exists():
            continue
        try:
            data = json.loads(run_file.read_text())
            model_obj = data.get("model", {}) or {}
            machine = data.get("machine", {}) or {}
            speed_metrics = data.get("speed_metrics", {}) or {}
            suite_map = data.get("suites", {}) or {}
            suite_names = list(suite_map.keys())
            is_full = REQUIRED_FULL_SUITES.issubset(set(suite_names))

            # Recompute speed score from tok/s using the v2 curve so historical
            # runs (which used the old 25*log2 capped-at-100 curve) display
            # comparably to new runs.
            gen_tok_per_sec = speed_metrics.get("generation_tok_per_sec", 0) or 0
            recomputed_speed = _recompute_speed_score(gen_tok_per_sec)
            speed_score_v2 = recomputed_speed
            # Recompute overall using new speed score (quality/reliability unchanged).
            quality_v2 = data.get("quality_score", 0)
            reliability_v2 = data.get("reliability_score", 0)
            overall_v2 = 0.55 * quality_v2 + 0.20 * speed_score_v2 + 0.25 * reliability_v2
            runs.append({
                "id": d.name,
                "timestamp": data.get("timestamp", ""),
                "model": model_obj.get("model_id", "unknown"),
                "quantization": model_obj.get("quantization", "") or "",
                "family": model_obj.get("family", "") or "",
                "parameter_count": model_obj.get("parameter_count", "") or "",
                "overall_score": overall_v2,
                "quality_score": quality_v2,
                "speed_score": speed_score_v2,
                "reliability_score": reliability_v2,
                "overall_score_raw": data.get("overall_score", 0),
                "speed_score_raw": data.get("speed_score", 0),
                "value_score": data.get("value_score", 0),
                "total_runtime_sec": data.get("total_runtime_sec", 0),
                "harness": data.get("harness", "raw"),
                "suites": {
                    name: {
                        "score": s.get("score", 0),
                        "pass_count": s.get("pass_count", 0),
                        "task_count": s.get("task_count", 0),
                    }
                    for name, s in suite_map.items()
                },
                "suite_count": len(suite_names),
                "suite_names": suite_names,
                "is_full_benchmark": is_full,
                "provider": data.get("provider", ""),
                "backend": machine.get("backend", data.get("provider", "")),
                "machine": machine.get("machine_id", ""),
                "gpu": machine.get("gpu", ""),
                "gpu_memory_gb": machine.get("gpu_memory_gb", 0),
                "cpu": machine.get("cpu", ""),
                "system_memory_gb": machine.get("system_memory_gb", 0),
                "os": machine.get("os", ""),
                "generation_tok_per_sec": speed_metrics.get("generation_tok_per_sec", 0),
                "prompt_eval_tok_per_sec": speed_metrics.get("prompt_eval_tok_per_sec", 0),
                "ttft_ms": speed_metrics.get("ttft_ms", 0),
            })
        except Exception:
            continue

    return {"runs": runs}


@router.get("/benchmark/runs/{run_id}")
async def get_run(run_id: str):
    """Get detailed run result."""
    # Check active runs first
    if run_id in _active_runs:
        return _active_runs[run_id]

    # Check disk
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        return {"error": "Run not found"}

    run_file = run_dir / "run.json"
    if not run_file.exists():
        return {"error": "Run data missing"}

    data = json.loads(run_file.read_text())

    hw_file = run_dir / "hardware.json"
    hw_data = None
    if hw_file.exists():
        hw_data = json.loads(hw_file.read_text())

    return {
        "status": "completed",
        "result": data,
        "hardware": hw_data,
    }
