"""Persist benchmark results."""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from rich.console import Console

from bench_loop.models import BenchmarkRun


RUNS_DIR = Path.home() / ".bench-loop" / "runs"


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


def save_run(run: BenchmarkRun, endpoint: str | None = None, console: Console | None = None) -> Path:
    console = console or Console()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    endpoint_id = _endpoint_identifier(endpoint)
    run_dir = RUNS_DIR / f"{timestamp}-{_slugify(run.model.model_id)}-{endpoint_id}-{_slugify(run.provider)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    output_path = run_dir / "run.json"
    output_path.write_text(json.dumps(run.to_dict(), indent=2), encoding="utf-8")
    console.print(f"Saved results to [bold]{output_path}[/bold]")
    return output_path
