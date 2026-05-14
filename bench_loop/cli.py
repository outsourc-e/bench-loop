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
    help="Harness adapter to use (e.g. raw, hermes, qwen, pi).",
)
@click.option(
    "--hardware",
    default=None,
    help="Hardware label stamped on the run (e.g. 'NVIDIA RTX 4090 24GB'). Useful when benchmarking through a tunnel.",
)
@click.option(
    "--gpu",
    default=None,
    help="GPU name override (e.g. 'NVIDIA RTX 4090').",
)
@click.option(
    "--gpu-memory-gb",
    default=None,
    type=float,
    help="GPU memory in GB to stamp on the run.",
)
@click.option(
    "--profile-name",
    default=None,
    help="Optional public display name attached to published runs.",
)
@click.option(
    "--profile-avatar-url",
    default=None,
    help="Optional avatar URL attached to published runs.",
)
@click.option(
    "--profile-url",
    default=None,
    help="Optional profile URL attached to published runs.",
)
@click.option(
    "--command-used",
    default=None,
    help="Optional launch command or config snippet to publish alongside the run.",
)
def run(
    model: str,
    endpoint: str,
    provider: str,
    suites: str,
    harness: str,
    hardware: str | None,
    gpu: str | None,
    gpu_memory_gb: float | None,
    profile_name: str | None,
    profile_avatar_url: str | None,
    profile_url: str | None,
    command_used: str | None,
) -> None:
    """Run a benchmark."""
    # Auto-detect provider for common ports if the user left it as the default
    # 'ollama' but pointed at an OpenAI-compatible server. LM Studio (:1234),
    # vLLM (:8000), llama.cpp's server (:8080), Jan (:1337), Osaurus, oMLX.
    if provider == "ollama":
        try:
            from urllib.parse import urlparse
            port = urlparse(endpoint).port
            if port in {1234, 1337, 5001, 8000, 8080, 8081}:
                provider = "openai_compat"
                click.echo(f"[auto-detected provider=openai_compat from port {port}]", err=True)
        except Exception:
            pass

    # Surface CLI hardware overrides to detect_hardware() via env vars so the
    # whole detection pipeline picks them up without threading another arg.
    if hardware:
        os.environ["BENCHLOOP_HARDWARE_LABEL"] = hardware
    if gpu:
        os.environ["BENCHLOOP_GPU"] = gpu
    if gpu_memory_gb is not None:
        os.environ["BENCHLOOP_GPU_MEMORY_GB"] = str(gpu_memory_gb)

    publish_profile = {
        "name": profile_name,
        "avatar_url": profile_avatar_url,
        "profile_url": profile_url,
    }
    command_used = (command_used or os.environ.get("BENCHLOOP_COMMAND_USED") or "").strip() or None

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
        # Catch-all so a single mid-run failure (HTTP 500, OOM, network hiccup,
        # ConnectError, ReadTimeout) gives the user a clean error instead of a
        # Python traceback.
        type_name = type(exc).__name__
        msg = str(exc)
        click.secho(f"\n✗ Benchmark failed ({type_name}): {msg or 'no message'}\n", fg="red", err=True)
        if type_name in {"ConnectError", "ConnectionRefusedError", "ConnectionError"} or "connection" in msg.lower():
            click.echo(f"Could not reach endpoint {endpoint}.", err=True)
            click.echo("Tips:", err=True)
            click.echo("  • Start your local LLM server:", err=True)
            click.echo("      Ollama:    ollama serve", err=True)
            click.echo("      LM Studio: launch app, enable local server", err=True)
            click.echo("  • Verify the endpoint URL and port are right.", err=True)
            click.echo("  • If your endpoint isn't Ollama, pass --provider openai_compat", err=True)
        elif "500" in msg or "Internal Server Error" in msg:
            click.echo("The provider returned HTTP 500. Common causes:", err=True)
            click.echo("  • Model context window exceeded for this prompt", err=True)
            click.echo("  • GPU OOM (try a smaller model or close other models)", err=True)
            click.echo("  • Ollama crashed (check `ollama serve` logs)", err=True)
        elif "timeout" in msg.lower() or type_name == "ReadTimeout":
            click.echo("Timeout. Try a smaller model, fewer suites, or check network stability.", err=True)
        raise SystemExit(1)
    print_run_report(benchmark)
    save_run(
        benchmark,
        endpoint=endpoint,
        publish_profile=publish_profile,
        command_used=command_used,
    )


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
            "profile_name": ((data.get("profile") or {}).get("name") or ""),
            "profile_avatar_url": ((data.get("profile") or {}).get("avatar_url") or ""),
            "profile_url": ((data.get("profile") or {}).get("profile_url") or ""),
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
@click.option("--port", "port", default=8877, show_default=True, type=int, help="Port for the dashboard (API + UI).")
@click.option("--api-port", default=None, type=int, help="DEPRECATED. Same as --port; kept for compatibility.")
@click.option("--ui-port", default=None, type=int, help="DEPRECATED. UI is now served by the API.")
@click.option("--api-only", is_flag=True, help="Legacy flag, no-op now that UI is bundled.")
@click.option("--dev/--no-dev", default=False, help="Use the sibling bench-loop-web repo with hot-reload (developer mode).")
@click.option(
    "--service-template",
    type=click.Choice(["launchd", "systemd", "windows-task"], case_sensitive=False),
    default=None,
    help="Print a persistence template instead of launching the dashboard.",
)
def dashboard(host: str, port: int, api_port: int | None, ui_port: int | None, api_only: bool, dev: bool, service_template: str | None) -> None:
    """Launch the local web dashboard.

    By default this runs the bundled FastAPI + React app that ships inside the
    benchloop-cli wheel — no extra clone or `make dev` needed.

    Pass --dev to attach to a sibling `bench-loop-web/` repo for hot-reload
    (Vite UI + uvicorn --reload).
    """
    import shutil
    import subprocess
    import webbrowser

    if api_port is not None:
        port = api_port  # back-compat
    _ = ui_port  # ignored; bundled mode serves UI on `port`
    _ = api_only  # ignored; bundled mode is single-process

    if service_template:
        command = f"benchloop dashboard --host {host} --port {port}"
        if service_template == "launchd":
            click.echo(f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <dict>
    <key>Label</key><string>com.benchloop.dashboard</string>
    <key>ProgramArguments</key>
    <array>
      <string>/bin/sh</string>
      <string>-lc</string>
      <string>{command}</string>
    </array>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>StandardOutPath</key><string>~/Library/Logs/benchloop-dashboard.log</string>
    <key>StandardErrorPath</key><string>~/Library/Logs/benchloop-dashboard.err</string>
  </dict>
</plist>''')
        elif service_template == "systemd":
            click.echo(f'''[Unit]
Description=BenchLoop dashboard
After=network.target

[Service]
Type=simple
ExecStart=/bin/sh -lc '{command}'
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target''')
        else:
            click.echo(f'''# PowerShell Scheduled Task / startup command
# Run this in a persistent shell or wrap it in Task Scheduler:
{command}

# Example Task Scheduler action:
# Program/script: powershell.exe
# Add arguments: -NoProfile -WindowStyle Hidden -Command "{command}"''')
        return

    if dev:
        web_dir = Path(os.environ.get(
            "BENCHLOOP_WEB_DIR",
            Path(__file__).resolve().parent.parent.parent / "bench-loop-web",
        )).resolve()
        api_dir = web_dir / "api"
        ui_dir = web_dir / "ui"
        if not api_dir.is_dir():
            click.echo(
                f"--dev mode: bench-loop-web not found at {api_dir}.\n"
                "Clone https://github.com/outsourc-e/bench-loop-web alongside this repo,\n"
                "or set $BENCHLOOP_WEB_DIR.",
                err=True,
            )
            sys.exit(1)
        env = os.environ.copy()
        env["BENCH_LOOP_DIR"] = str(Path(__file__).resolve().parent.parent)
        env["PYTHONPATH"] = env["BENCH_LOOP_DIR"] + os.pathsep + env.get("PYTHONPATH", "")
        api_proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "main:app",
             "--host", host, "--port", str(port), "--app-dir", str(api_dir), "--reload"],
            env=env,
        )
        click.echo(f"BenchLoop API (dev):  http://{host}:{port}")
        ui_proc = None
        if ui_dir.is_dir() and shutil.which("npx"):
            ui_dev_port = ui_port or 5180
            ui_proc = subprocess.Popen(
                ["npx", "vite", "--host", host, "--port", str(ui_dev_port)],
                cwd=str(ui_dir),
                env=env,
            )
            click.echo(f"BenchLoop UI (vite):  http://{host}:{ui_dev_port}")
    else:
        # Bundled mode — use the assets shipped in the wheel.
        api_dir = Path(__file__).resolve().parent / "dashboard" / "api"
        ui_dir = Path(__file__).resolve().parent / "dashboard" / "ui"
        if not api_dir.is_dir() or not (ui_dir / "index.html").exists():
            click.echo(
                "Bundled dashboard assets not found. Reinstall benchloop-cli, "
                "or use `benchloop dashboard --dev` against a checkout.",
                err=True,
            )
            sys.exit(1)
        env = os.environ.copy()
        env["BENCH_LOOP_DIR"] = str(Path(__file__).resolve().parent.parent)
        env["PYTHONPATH"] = env["BENCH_LOOP_DIR"] + os.pathsep + env.get("PYTHONPATH", "")
        api_proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "main:app",
             "--host", host, "--port", str(port), "--app-dir", str(api_dir)],
            env=env,
        )
        url = f"http://{host}:{port}"
        click.echo(f"BenchLoop dashboard: {url}")
        try:
            webbrowser.open(url, new=2)
        except Exception:
            pass
        ui_proc = None

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
