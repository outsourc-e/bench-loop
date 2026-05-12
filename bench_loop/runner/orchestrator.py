"""Benchmark orchestrator."""
from __future__ import annotations

import time
from dataclasses import fields
from datetime import datetime, timezone
from statistics import median
from typing import Any
from urllib.parse import urlparse

from bench_loop.harness import get_harness
from bench_loop.hardware import detect_hardware
from bench_loop.models import BenchmarkRun, MachineInfo, ModelInfo, SpeedMetrics, SuiteResult, TaskResult
from bench_loop.providers import ollama
from bench_loop.suites import (
    DEFAULT_SUITES as DEFAULT_SUITES,
    SUITE_REGISTRY as SUITE_REGISTRY,
)
from bench_loop.suites.speed import SpeedSuite

from bench_loop.providers import openai_compat
PROVIDER_REGISTRY = {
    "ollama": ollama,
    "openai": openai_compat,
    "openai_compat": openai_compat,
    "vmlx": openai_compat,  # vmlx exposes OpenAI-compatible /v1
}
SPEED_TRIALS = 3


async def run_benchmark(
    *args,
    model: str | None = None,
    endpoint: str | None = None,
    provider: str = "ollama",
    suites: list[str] | None = None,
    suite_names: list[str] | None = None,  # alias for API back-compat
    harness: str = "raw",
    on_progress=None,
    runs: int | None = None,  # accepted but currently unused (single-run)
    timeout_sec: float | None = None,  # accepted but unused
) -> BenchmarkRun:
    # API back-compat: allow `run_benchmark(config)` where config has
    # the same attributes (model/endpoint/provider/suite_names/harness/...).
    if args and not (model or endpoint):
        cfg = args[0]
        model = getattr(cfg, "model", None) or model
        endpoint = getattr(cfg, "endpoint", None) or getattr(cfg, "base_url", None) or endpoint
        provider = getattr(cfg, "provider", None) or provider
        cfg_suites = getattr(cfg, "suite_names", None) or getattr(cfg, "suites", None)
        suites = cfg_suites or suites
        harness = getattr(cfg, "harness", None) or harness
    elif suite_names and not suites:
        suites = suite_names

    if not model or not endpoint:
        raise ValueError("run_benchmark requires both `model` and `endpoint`")
    if provider not in PROVIDER_REGISTRY:
        raise ValueError(f"Unsupported provider: {provider}")

    provider_module = PROVIDER_REGISTRY[provider]
    selected_suites = suites or DEFAULT_SUITES

    hardware = detect_hardware(endpoint=endpoint)
    machine_kwargs = {
        field.name: hardware.get(field.name, field.default)
        for field in fields(MachineInfo)
        if field.init
    }
    machine = MachineInfo(**machine_kwargs)

    system_info: dict[str, Any] = {}
    if hasattr(provider_module, "get_system_info"):
        system_info = await provider_module.get_system_info(endpoint)

    endpoint_host = _endpoint_host(endpoint)
    if endpoint_host and endpoint_host not in {"localhost", "127.0.0.1", "::1"}:
        remote_label = system_info.get("endpoint") or endpoint
        machine.machine_id = f"{machine.machine_id} ({endpoint_host})"
        machine.backend = f"{provider}:{remote_label}"
    elif not machine.backend:
        machine.backend = provider

    available_models = await provider_module.list_models(endpoint)
    if model not in available_models:
        raise ValueError(f"Model '{model}' not found on {endpoint}. Available: {', '.join(available_models)}")

    run_started = time.perf_counter()
    await provider_module.chat(
        endpoint=endpoint,
        model=model,
        messages=[{"role": "user", "content": "Reply with: warmup"}],
        max_tokens=8,
        temperature=0.0,
    )

    harness_adapter = get_harness(harness)

    run = BenchmarkRun(
        timestamp=datetime.now(timezone.utc).isoformat(),
        model=ModelInfo(model_id=model),
        machine=machine,
        provider=provider,
        harness=harness,
        harness_version=getattr(harness_adapter, 'version', ''),
    )

    speed_metric_samples: list[SpeedMetrics] = []

    # Pre-compute suite_task_counts for live API consumers.
    total_tasks_all = 0
    suite_task_counts: dict[str, int] = {}
    for sn in selected_suites:
        if sn in SUITE_REGISTRY:
            try:
                _tasks_preview = await SUITE_REGISTRY[sn]().load_tasks()
                suite_task_counts[sn] = len(_tasks_preview)
                total_tasks_all += len(_tasks_preview)
            except Exception:
                suite_task_counts[sn] = 0
    if on_progress:
        try:
            on_progress({
                "type": "run_started",
                "total_tasks": total_tasks_all,
                "suites": list(selected_suites),
                "suite_task_counts": suite_task_counts,
            })
        except Exception:
            pass

    completed_so_far = 0
    for suite_name in selected_suites:
        if suite_name not in SUITE_REGISTRY:
            raise ValueError(f"Unknown suite: {suite_name}")
        suite = SUITE_REGISTRY[suite_name]()
        tasks = await suite.load_tasks()
        if on_progress:
            try:
                on_progress({"type": "suite_started", "suite": suite_name, "task_count": len(tasks)})
            except Exception:
                pass
        task_results: list[TaskResult] = []
        for task in tasks:
            if suite_name == "speed":
                result = await _run_speed_task(
                    provider_module,
                    endpoint,
                    model,
                    suite,
                    task,
                    harness=harness_adapter,
                    provider_name=provider,
                )
            else:
                result = await suite.run_task(
                    provider_module,
                    endpoint,
                    model,
                    task,
                    harness=harness_adapter,
                    provider_name=provider,
                )
            task_results.append(result)
            speed_meta = result.metadata.get("speed_metrics") if isinstance(result.metadata, dict) else None
            if isinstance(speed_meta, dict):
                speed_metric_samples.append(SpeedMetrics(**speed_meta))
            completed_so_far += 1
            if on_progress:
                try:
                    on_progress({
                        "type": "task_completed",
                        "suite": suite_name,
                        "task_id": result.task_id,
                        "score": result.score,
                        "passed": result.passed,
                        "latency_ms": result.latency_ms,
                        "error": result.error,
                        "completed_tasks": completed_so_far,
                        "total_tasks": total_tasks_all,
                    })
                except Exception:
                    pass

        latencies = [task.latency_ms for task in task_results if task.latency_ms > 0]
        score = suite.aggregate_score(task_results)
        pass_count = sum(1 for task in task_results if task.passed)
        suite_result = SuiteResult(
            suite=suite_name,
            score=score,
            task_count=len(task_results),
            pass_count=pass_count,
            fail_count=len(task_results) - pass_count,
            median_latency_ms=median(latencies) if latencies else 0.0,
            tasks=task_results,
        )
        run.suites[suite_name] = suite_result
        if on_progress:
            try:
                on_progress({
                    "type": "suite_completed",
                    "suite": suite_name,
                    "score": suite_result.score,
                    "pass_count": suite_result.pass_count,
                    "task_count": suite_result.task_count,
                })
            except Exception:
                pass

    run.total_runtime_sec = time.perf_counter() - run_started
    if speed_metric_samples:
        run.speed_metrics = SpeedMetrics(
            ttft_ms=sum(item.ttft_ms for item in speed_metric_samples) / len(speed_metric_samples),
            prompt_eval_tok_per_sec=sum(item.prompt_eval_tok_per_sec for item in speed_metric_samples)
            / len(speed_metric_samples),
            generation_tok_per_sec=sum(item.generation_tok_per_sec for item in speed_metric_samples)
            / len(speed_metric_samples),
            total_latency_ms=sum(item.total_latency_ms for item in speed_metric_samples)
            / len(speed_metric_samples),
        )
    run.compute_aggregates()
    if on_progress:
        try:
            on_progress({
                "type": "run_completed",
                "overall_score": run.overall_score,
                "quality_score": run.quality_score,
                "speed_score": run.speed_score,
                "reliability_score": run.reliability_score,
                "total_runtime_sec": run.total_runtime_sec,
            })
        except Exception:
            pass
    return run


async def _run_speed_task(
    provider_module: Any,
    endpoint: str,
    model: str,
    suite: SpeedSuite,
    task: Any,
    harness: Any | None = None,
    provider_name: str = "ollama",
) -> TaskResult:
    trial_results: list[TaskResult] = []
    request = (
        harness.prepare(task, provider_name=provider_name)
        if harness is not None
        else {"messages": task.messages, **task.config}
    )
    for _ in range(SPEED_TRIALS):
        response = await provider_module.chat(
            endpoint=endpoint,
            model=model,
            **request,
        )
        if harness is not None:
            response = harness.postprocess(response, task)
        trial_results.append(suite.evaluate(task, response))

    scored_trials = trial_results[1:] if len(trial_results) > 1 else trial_results
    selected = min(scored_trials, key=lambda item: abs(item.score - median(t.score for t in scored_trials)))
    selected.metadata = {
        **selected.metadata,
        "trials": [
            {
                "trial_index": index + 1,
                "warmup": index == 0,
                "passed": result.passed,
                "score": result.score,
                "latency_ms": result.latency_ms,
                "tokens_generated": result.tokens_generated,
                "tokens_prompt": result.tokens_prompt,
                "error": result.error,
                "speed_metrics": result.metadata.get("speed_metrics", {}),
            }
            for index, result in enumerate(trial_results)
        ],
        "selected_trial": next(
            index + 1 for index, result in enumerate(trial_results) if result is selected
        ),
        "warmup_dropped": len(trial_results) > 1,
        "selection_method": "median_of_post_warmup_trials",
    }
    return selected



def _endpoint_host(endpoint: str) -> str:
    parsed = urlparse(endpoint if "://" in endpoint else f"http://{endpoint}")
    return (parsed.hostname or "").strip().lower()
