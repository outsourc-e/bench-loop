"""BenchLoop CLI."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import click

from bench_loop import __version__
from bench_loop.harness import list_harnesses
from bench_loop.report.console import print_run_report
from bench_loop.runner.orchestrator import DEFAULT_SUITES, SUITE_REGISTRY, run_benchmark
from bench_loop.runner.result_writer import save_run

RUNS_DIR = Path(os.environ.get("BENCHLOOP_RUNS", Path.home() / ".bench-loop" / "runs")).expanduser()


@click.group(help="BenchLoop local LLM benchmarking CLI.")
@click.version_option(version=__version__, prog_name="bench-loop")
def main() -> None:
    """BenchLoop local LLM benchmarking CLI."""


@dataclass
class SuiteSummary:
    name: str
    task_count: int


async def _suite_summaries() -> list[SuiteSummary]:
    summaries: list[SuiteSummary] = []
    for suite_name, suite_cls in SUITE_REGISTRY.items():
        suite = suite_cls()
        tasks = await suite.load_tasks()
        summaries.append(SuiteSummary(name=suite.name, task_count=len(tasks)))
    summaries.sort(key=lambda item: item.name)
    return summaries


@main.command()
def info() -> None:
    """Show installed suites, harnesses, and fixture counts."""
    summaries = asyncio.run(_suite_summaries())
    click.echo(f"BenchLoop v{__version__}")
    click.echo("\nSupported suites:")
    for summary in summaries:
        click.echo(f"  {summary.name}: {summary.task_count} tasks")
    click.echo("\nAvailable harnesses:")
    for harness_name in list_harnesses():
        click.echo(f"  {harness_name}")


@main.command()
@click.option("--model", required=True, help="Model name to benchmark.")
@click.option("--endpoint", default="http://localhost:11434", show_default=True, help="Provider endpoint URL.")
@click.option("--provider", default="ollama", show_default=True, help="Provider backend.")
@click.option(
    "--suites",
    default=",".join(DEFAULT_SUITES),
    show_default=True,
    help="Comma-separated suite list.",
)
@click.option(
    "--harness",
    default="raw",
    show_default=True,
    help="Harness adapter to use (e.g. raw, hermes).",
)
def run(model: str, endpoint: str, provider: str, suites: str, harness: str) -> None:
    """Run a benchmark."""
    selected_suites = [item.strip() for item in suites.split(",") if item.strip()]
    try:
        benchmark = asyncio.run(
            run_benchmark(
                model=model,
                endpoint=endpoint,
                provider=provider,
                suites=selected_suites,
                harness=harness,
            )
        )
    except ValueError as exc:
        msg = str(exc)
        click.secho(f"\n✗ {msg}\n", fg="red", err=True)
        if "not found on" in msg:
            click.echo("Tips:", err=True)
            click.echo("  • Make sure your local LLM server is running:", err=True)
            click.echo("      Ollama:    ollama serve", err=True)
            click.echo("      LM Studio: launch app, enable local server", err=True)
            click.echo("  • Pull a model first, e.g.:", err=True)
            click.echo("      ollama pull qwen3:1.7b", err=True)
            click.echo("  • If your endpoint isn't Ollama, pass --provider openai_compat", err=True)
            click.echo("  • Or launch the dashboard which auto-discovers models:", err=True)
            click.echo("      benchloop dashboard", err=True)
        raise SystemExit(1)
    except ConnectionError as exc:
        click.secho(f"\n✗ Could not reach endpoint {endpoint}: {exc}\n", fg="red", err=True)
        click.echo("Is your local LLM server running?", err=True)
        raise SystemExit(1)
    except Exception as exc:  # noqa: BLE001
        # Catch-all so a single mid-run failure (HTTP 500, OOM, network hiccup)
        # gives the user a clean error instead of a Python traceback.
        type_name = type(exc).__name__
        click.secho(f"\n✗ Benchmark failed ({type_name}): {exc}\n", fg="red", err=True)
        if "500" in str(exc) or "Internal Server Error" in str(exc):
            click.echo("The provider returned HTTP 500. Common causes:", err=True)
            click.echo("  • Model context window exceeded for this prompt", err=True)
            click.echo("  • GPU OOM (try a smaller model or close other models)", err=True)
            click.echo("  • Ollama crashed (check `ollama serve` logs)", err=True)
        elif "timeout" in str(exc).lower():
            click.echo("Timeout. Try a smaller model or check network stability.", err=True)
        click.echo("\nFull error type: " + type_name, err=True)
        raise SystemExit(1)
    print_run_report(benchmark)
    save_run(benchmark, endpoint=endpoint)


@main.command()
def suites() -> None:
    """List available benchmark suites."""
    summaries = asyncio.run(_suite_summaries())
    for summary in summaries:
        click.echo(f"{summary.name}: {summary.task_count} tasks")


@main.command()
@click.option("--output", "-o", default=None, help="Path to write the leaderboard JSON. Defaults to stdout.")
@click.option("--all", "include_all", is_flag=True, help="Include partial runs (default: only full benchmarks).")
def export(output: str | None, include_all: bool) -> None:
    """Export local runs to a leaderboard-compatible JSON.

    The output format matches the schema consumed by https://bench-loop.com/leaderboard
    so you can submit your own runs via PR.
    """
    REQUIRED_FULL = {"speed", "toolcall", "dataextract", "instructfollow", "reasonmath"}
    REQUIRED_QUALITY = {"toolcall", "dataextract", "instructfollow", "reasonmath"}

    if not RUNS_DIR.exists():
        click.echo(f"No runs found at {RUNS_DIR}", err=True)
        sys.exit(1)

    rows: dict[str, dict] = {}
    for run_dir in sorted(RUNS_DIR.iterdir(), reverse=True):
        run_file = run_dir / "run.json"
        if not run_file.is_file():
            continue
        try:
            data = json.loads(run_file.read_text())
        except Exception as exc:
            click.echo(f"skipped {run_dir.name}: {exc}", err=True)
            continue

        suite_map = data.get("suites") or {}
        suite_names = set(suite_map.keys())
        is_full = REQUIRED_FULL.issubset(suite_names)
        is_quality = REQUIRED_QUALITY.issubset(suite_names)
        is_agent_only = suite_names == {"agent"}

        if not include_all and not (is_full or is_quality or is_agent_only):
            continue

        model_id = (data.get("model") or {}).get("model_id", "unknown")
        if "/" in model_id and model_id.endswith(".gguf"):
            model_id = model_id.split("/")[-1]

        row = {
            "id": run_dir.name,
            "timestamp": data.get("timestamp", ""),
            "model": model_id,
            "harness": data.get("harness", "raw"),
            "provider": data.get("provider", ""),
            "machine": (data.get("machine") or {}).get("gpu")
                or (data.get("machine") or {}).get("cpu")
                or (data.get("machine") or {}).get("machine_id", ""),
            "overall_score": data.get("overall_score", 0),
            "quality_score": data.get("quality_score", 0),
            "speed_score": data.get("speed_score", 0),
            "reliability_score": data.get("reliability_score", 0),
            "generation_tok_per_sec": (data.get("speed_metrics") or {}).get("generation_tok_per_sec", 0),
            "ttft_ms": (data.get("speed_metrics") or {}).get("ttft_ms", 0),
            "is_full_benchmark": is_full,
            "is_quality_full": is_quality,
            "is_agent_only": is_agent_only,
            "agent_score": (suite_map.get("agent") or {}).get("score"),
            "agent_pass": (suite_map.get("agent") or {}).get("pass_count"),
            "agent_task_count": (suite_map.get("agent") or {}).get("task_count"),
            "suites": {name: {"score": s.get("score", 0)} for name, s in suite_map.items()},
        }

        # Keep best run per model+harness.
        key = f"{model_id}::{row['harness']}"
        existing = rows.get(key)
        if not existing or row["overall_score"] > existing["overall_score"]:
            rows[key] = row

    payload = {
        "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "count": len(rows),
        "source": "benchloop export",
        "runs": sorted(rows.values(), key=lambda r: r["overall_score"], reverse=True),
    }

    if output:
        Path(output).write_text(json.dumps(payload, indent=2))
        click.echo(f"Wrote {len(rows)} runs to {output}")
    else:
        click.echo(json.dumps(payload, indent=2))


@main.command()
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--api-port", default=8877, show_default=True, type=int)
@click.option("--ui-port", default=5180, show_default=True, type=int)
@click.option("--api-only", is_flag=True, help="Only start the API; serve the UI yourself.")
def dashboard(host: str, api_port: int, ui_port: int, api_only: bool) -> None:
    """Launch the local web dashboard.

    Looks for the BenchLoop web app at $BENCHLOOP_WEB_DIR (default: ../bench-loop-web).
    If the bundled web app cannot be located, just run the API and tell the user
    where to clone the web app.
    """
    import shutil
    import subprocess

    web_dir = Path(os.environ.get(
        "BENCHLOOP_WEB_DIR",
        Path(__file__).resolve().parent.parent.parent / "bench-loop-web",
    )).resolve()
    api_dir = web_dir / "api"
    ui_dir = web_dir / "ui"

    if not api_dir.is_dir():
        click.echo(
            f"Web API not found at {api_dir}.\n"
            "Clone https://github.com/outsourc-e/bench-loop-web alongside this repo,\n"
            "or set $BENCHLOOP_WEB_DIR.",
            err=True,
        )
        sys.exit(1)

    env = os.environ.copy()
    env["BENCH_LOOP_DIR"] = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = env["BENCH_LOOP_DIR"] + os.pathsep + env.get("PYTHONPATH", "")

    api_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "main:app",
            "--host", host,
            "--port", str(api_port),
            "--app-dir", str(api_dir),
        ],
        env=env,
    )
    click.echo(f"BenchLoop API:  http://{host}:{api_port}")

    ui_proc = None
    if not api_only and ui_dir.is_dir() and shutil.which("npx"):
        ui_proc = subprocess.Popen(
            ["npx", "vite", "--host", host, "--port", str(ui_port)],
            cwd=str(ui_dir),
            env=env,
        )
        click.echo(f"BenchLoop UI:   http://{host}:{ui_port}")
    elif not api_only:
        click.echo(f"(UI skipped — vite/npx not found in PATH; run `npm install && npm run dev` in {ui_dir})")

    try:
        api_proc.wait()
    except KeyboardInterrupt:
        click.echo("\nShutting down dashboard...")
    finally:
        api_proc.terminate()
        if ui_proc:
            ui_proc.terminate()


if __name__ == "__main__":
    main()
