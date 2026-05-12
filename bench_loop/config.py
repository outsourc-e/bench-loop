"""BenchLoop configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


BENCH_LOOP_DIR = Path(__file__).parent
TASKS_DIR = BENCH_LOOP_DIR / "tasks"
RESULTS_DIR = Path.cwd() / "results"


@dataclass
class RunConfig:
    model: str = ""
    provider: str = "ollama"
    harness: str = "raw"
    suites: list[str] = field(default_factory=list)
    trials: int = 3
    warmup: bool = True
    output_dir: str = "results"
    base_url: str = "http://localhost:11434"
