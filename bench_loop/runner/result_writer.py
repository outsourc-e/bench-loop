"""Persist benchmark results."""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import httpx
from rich.console import Console

from bench_loop.hardware import detect_hardware
from bench_loop.models import BenchmarkRun


RUNS_DIR = Path.home() / ".bench-loop" / "runs"

# Public leaderboard submit endpoint. Set BENCHLOOP_NO_SUBMIT=1 to disable.
LEADERBOARD_SUBMIT_URL = os.environ.get(
    "BENCHLOOP_SUBMIT_URL", "https://api.bench-loop.com/submit"
)
_SUBMIT_DISABLED = os.environ.get("BENCHLOOP_NO_SUBMIT", "").lower() in {"1", "true", "yes"}


def _coalesce_profile(publish_profile: dict | None = None) -> dict[str, str]:
    raw = {
        "name": (publish_profile or {}).get("name") or os.environ.get("BENCHLOOP_PROFILE_NAME", ""),
        "avatar_url": (publish_profile or {}).get("avatar_url") or os.environ.get("BENCHLOOP_PROFILE_AVATAR_URL", ""),
        "profile_url": (publish_profile or {}).get("profile_url") or os.environ.get("BENCHLOOP_PROFILE_URL", ""),
    }
    return {key: str(value).strip() for key, value in raw.items() if str(value or "").strip()}


def _submit_to_leaderboard(payload: dict, console: Console) -> None:
    """Submit to public leaderboard. Short timeout, never raises.

    Runs synchronously so the CLI doesn't exit before the HTTP request
    completes. Total worst-case added latency: 5s on network failure.
    """
    if _SUBMIT_DISABLED:
        return

    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.post(LEADERBOARD_SUBMIT_URL, json=payload)
            if resp.status_code == 200:
                console.print(
                    f"[dim green]→ published to https://bench-loop.com/leaderboard[/dim green]"
                )
            else:
                console.print(
                    f"[dim yellow]Leaderboard submit returned {resp.status_code}: {resp.text[:120]}[/dim yellow]"
                )
    except Exception as e:  # noqa: BLE001
        console.print(
            f"[dim yellow]Leaderboard submit skipped (offline?): {type(e).__name__}[/dim yellow]"
        )


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    return slug.strip("-") or "run"


def _endpoint_identifier(endpoint: str | None) -> str:
    if not endpoint:
        return "local"
    parsed = urlparse(endpoint if "://" in endpoint else f"http://{endpoint}")
    host = (parsed.hostname or "").strip().lower()
    if host in {"", "localhost", "127.0.0.1", "::1"}:
        return "local"
    if re.fullmatch(r"\d+\.\d+\.\d+\.\d+", host):
        return f"remote-{host.split('.')[-1]}"
    return f"remote-{_slugify(host)}"


def save_run(
    run: BenchmarkRun,
    endpoint: str | None = None,
    console: Console | None = None,
    publish_profile: dict | None = None,
    command_used: str | None = None,
) -> Path:
    console = console or Console()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    endpoint_id = _endpoint_identifier(endpoint)
    run_dir = RUNS_DIR / f"{timestamp}-{_slugify(run.model.model_id)}-{endpoint_id}-{_slugify(run.provider)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    output_path = run_dir / "run.json"
    run_dict = run.to_dict()
    profile = _coalesce_profile(publish_profile)
    if profile:
        run_dict["profile"] = profile
    if command_used and str(command_used).strip():
        run_dict["command_used"] = str(command_used).strip()
    output_path.write_text(json.dumps(run_dict, indent=2), encoding="utf-8")
    console.print(f"Saved results to [bold]{output_path}[/bold]")

    # Add a stable run_id (folder name) so the leaderboard can dedupe properly.
    run_dict["run_id"] = run_dir.name
    _submit_to_leaderboard(run_dict, console)

    return output_path


def save_failed_run(
    *,
    run_id: str,
    model: str,
    endpoint: str,
    provider: str,
    harness: str,
    suites: list[str],
    error: str,
    traceback_text: str | None = None,
    events: list[dict] | None = None,
    publish_profile: dict | None = None,
    command_used: str | None = None,
    console: Console | None = None,
) -> Path:
    console = console or Console()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    endpoint_id = _endpoint_identifier(endpoint)
    run_dir = RUNS_DIR / f"{timestamp}-{_slugify(model)}-{endpoint_id}-{_slugify(provider)}-failed"
    run_dir.mkdir(parents=True, exist_ok=True)

    machine = detect_hardware(endpoint=endpoint)
    run_dict = {
        "run_id": run_id,
        "status": "failed",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": {
            "model_id": model,
            "family": "",
            "parameter_count": "",
            "quantization": "",
        },
        "machine": machine,
        "provider": provider,
        "harness": harness,
        "requested_suites": suites,
        "suites": {},
        "overall_score": 0,
        "quality_score": 0,
        "speed_score": 0,
        "reliability_score": 0,
        "value_score": 0,
        "speed_metrics": {},
        "total_runtime_sec": 0,
        "error": error,
        "traceback": traceback_text or "",
        "events": events or [],
    }
    profile = _coalesce_profile(publish_profile)
    if profile:
        run_dict["profile"] = profile
    if command_used and str(command_used).strip():
        run_dict["command_used"] = str(command_used).strip()

    output_path = run_dir / "run.json"
    output_path.write_text(json.dumps(run_dict, indent=2), encoding="utf-8")
    console.print(f"Saved failed run to [bold]{output_path}[/bold]")
    return output_path
