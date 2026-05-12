from __future__ import annotations

from typing import Any

from rich.console import Console

from .. import __version__
from ..models import BenchmarkRun, TaskResult

console = Console()
ACCENT = "green"
BOLD_ACCENT = "bold green"
DIM = "dim"
HEADER_WIDTH = 56


def _score_style(score: float) -> str:
    if score >= 80:
        return "bold green"
    if score >= 60:
        return "bold yellow"
    return "bold red"


def _score_badge(score: float) -> str:
    if score >= 80:
        return "✅"
    if score >= 60:
        return "⚠️"
    return "❌"


def _render_header() -> None:
    inner = f"  ⚡ BENCHLOOP · local LLM benchmarking v{__version__}"
    padded = inner.ljust(HEADER_WIDTH - 2)
    lines = [
        f"╔{'═' * (HEADER_WIDTH - 2)}╗",
        f"║{padded}║",
        f"╚{'═' * (HEADER_WIDTH - 2)}╝",
    ]
    for line in lines:
        console.print(f"[{BOLD_ACCENT}]{line}[/{BOLD_ACCENT}]")
    console.print()


def _info_row(label: str, value: str) -> None:
    console.print(f"  [{BOLD_ACCENT}]◉[/{BOLD_ACCENT}] [{ACCENT}]{label:<10}[/{ACCENT}] {value}")


def _rule(title: str | None = None) -> None:
    if title:
        pad = HEADER_WIDTH - len(title) - 6
        console.print(f"\n[{DIM}]─── {title} {'─' * max(pad, 4)}[/{DIM}]")
    else:
        console.print(f"[{DIM}]{'─' * HEADER_WIDTH}[/{DIM}]")


def _summary_rule(title: str | None = None) -> None:
    if title:
        pad = HEADER_WIDTH - len(title) - 6
        console.print(f"\n[{BOLD_ACCENT}]═══ {title} {'═' * max(pad, 4)}[/{BOLD_ACCENT}]")
    else:
        console.print(f"[{BOLD_ACCENT}]{'═' * HEADER_WIDTH}[/{BOLD_ACCENT}]")


def render_hardware_summary(snapshot: dict[str, Any]) -> None:
    _render_header()
    _info_row("HOST", str(snapshot.get("machine_id", "unknown")))
    _info_row("OS", str(snapshot.get("os", "")))
    _info_row("CPU", str(snapshot.get("cpu", "")))
    _info_row("RAM", f"{snapshot.get('system_memory_gb', 0)} GB")
    _info_row("GPU", str(snapshot.get("gpu", "n/a") or "n/a"))
    if snapshot.get("gpu_memory_gb"):
        _info_row("VRAM", f"{snapshot.get('gpu_memory_gb')} GB")
    if snapshot.get("backend"):
        _info_row("BACKEND", str(snapshot.get("backend")))
    console.print()


def render_suite_progress(suite_name: str, result: TaskResult) -> None:
    score = result.score
    badge = _score_badge(score)
    style = _score_style(score)
    console.print(
        f"  [{style}]{suite_name}:{result.task_id:<24}[/{style}] "
        f"score=[{style}]{score:.1f}[/{style}] latency={result.latency_ms:.0f}ms {badge}"
    )


def render_run_summary(run: BenchmarkRun) -> None:
    _render_header()
    _info_row("MODEL", run.model.model_id)
    _info_row("HARNESS", f"{run.harness} ({run.provider})")
    if run.machine.summary():
        _info_row("MACHINE", run.machine.summary())
    if run.speed_metrics.generation_tok_per_sec:
        _info_row("GEN TOK/S", f"{run.speed_metrics.generation_tok_per_sec:.2f}")
    console.print()

    _rule("Suite Results")
    console.print()
    for suite in run.suites.values():
        score = suite.score
        badge = _score_badge(score)
        style = _score_style(score)
        pass_rate = f"{suite.pass_count}/{suite.task_count}"
        lat = f"{suite.median_latency_ms:.0f}ms"
        console.print(
            f"  [{ACCENT}]{suite.suite:<16}[/{ACCENT}] "
            f"[{style}]{score:>5.1f}[/{style}] [{DIM}]{pass_rate:>5}  {lat:>7}[/{DIM}] {badge}"
        )

    _summary_rule("Summary")
    console.print()
    rows = [
        ("QUALITY", run.quality_score),
        ("SPEED", run.speed_score),
        ("RELIABILITY", run.reliability_score),
        ("VALUE", run.value_score),
    ]
    for label, val in rows:
        console.print(f"  [{ACCENT}]{label:<14}[/{ACCENT}][{DIM}]│[/{DIM}]  {val:.1f}")
    console.print(f"  [{DIM}]{'─' * 24}[/{DIM}]")
    overall = run.overall_score
    badge = _score_badge(overall)
    style = _score_style(overall)
    console.print(
        f"  [{BOLD_ACCENT}]OVERALL       [/{BOLD_ACCENT}][{DIM}]│[/{DIM}]  [{style}]{overall:.1f}[/{style}]  {badge}"
    )
    console.print(f"\n  [{DIM}]Runtime: {run.total_runtime_sec:.1f}s[/{DIM}]")
    console.print()


def print_run_report(run: BenchmarkRun) -> None:
    render_run_summary(run)
