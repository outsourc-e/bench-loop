"""Speed suite fixture loader and evaluation."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from bench_loop.config import TASKS_DIR
from bench_loop.models import BenchmarkTask, SpeedMetrics, TaskResult
from bench_loop.suites.base import BenchmarkSuite


class SpeedSuite(BenchmarkSuite):
    name = "speed"
    task_file = Path(TASKS_DIR) / "speed" / "tasks.yaml"

    def evaluate(self, task: BenchmarkTask, response: dict[str, Any]) -> TaskResult:
        generation_tok_per_sec = float(response.get("generation_tok_per_sec") or 0.0)
        prompt_tok_per_sec = float(response.get("prompt_eval_tok_per_sec") or 0.0)
        speed_metrics = SpeedMetrics(
            ttft_ms=float(response.get("ttft_ms") or 0.0),
            prompt_eval_tok_per_sec=prompt_tok_per_sec,
            generation_tok_per_sec=generation_tok_per_sec,
            total_latency_ms=float(response.get("total_ms") or 0.0),
        )
        # Speed score is anchored on real-world reference points so similar tok/s
        # produces similar scores across runs. Reference anchors (empirical M-series + RTX):
        #   5 tok/s  -> 30  (slow CPU / very large model)
        #   15 tok/s -> 50  (modest local inference)
        #   30 tok/s -> 70  (typical local)
        #   60 tok/s -> 85  (fast local)
        #   120 tok/s -> 95
        #   240 tok/s -> ~100 (high-end GPU)
        # Curve: 12.54 * log2(tok/s) + 0.9, fitted so:
        #   5 tok/s -> ~30, 15 -> ~50, 30 -> ~62, 60 -> ~75,
        #   120 -> ~87, 240 -> ~100.
        # This keeps 20 tok/s and 200 tok/s from both reading as "100".
        if generation_tok_per_sec <= 0:
            score = 0.0
        else:
            score = 12.54 * math.log2(generation_tok_per_sec) + 0.9
        score = min(100.0, max(0.0, score))
        passed = bool(response.get("content", "").strip())
        return self.build_result(
            task=task,
            passed=passed,
            score=round(score, 2),
            response=response,
            output=self.response_text(response),
            metadata={
                "speed_metrics": speed_metrics.__dict__,
                "eval_count": int(response.get("eval_count") or 0),
                "eval_duration": int(response.get("eval_duration") or 0),
                "prompt_eval_count": int(response.get("prompt_eval_count") or 0),
                "prompt_eval_duration": int(response.get("prompt_eval_duration") or 0),
                "load_duration": int(response.get("load_duration") or 0),
            },
        )
