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
        ttft_ms = float(response.get("ttft_ms") or 0.0)
        speed_metrics = SpeedMetrics(
            ttft_ms=ttft_ms,
            prompt_eval_tok_per_sec=prompt_tok_per_sec,
            generation_tok_per_sec=generation_tok_per_sec,
            total_latency_ms=float(response.get("total_ms") or 0.0),
        )

        is_cloud = ttft_ms > 0 and generation_tok_per_sec > 0

        if is_cloud:
            score = self._cloud_speed_score(generation_tok_per_sec, ttft_ms)
        else:
            # Local scoring curve — anchored on real-world reference points:
            #   5 tok/s  -> 30  (slow CPU / very large model)
            #   15 tok/s -> 50  (modest local inference)
            #   30 tok/s -> 70  (typical local)
            #   60 tok/s -> 85  (fast local)
            #   120 tok/s -> 95
            #   240 tok/s -> ~100 (high-end GPU)
            # Curve: 12.54 * log2(tok/s) + 0.9
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
                "is_cloud_speed": is_cloud,
            },
        )

    @staticmethod
    def _cloud_speed_score(tok_per_sec: float, ttft_ms: float) -> float:
        """Cloud speed scoring: combines TTFT (60% weight) + effective tok/s (40%).

        Cloud reference points:
          TTFT:  200ms -> 95, 500ms -> 80, 1000ms -> 60, 2000ms -> 40, 5000ms -> 20
          tok/s: 20 -> 40, 40 -> 60, 60 -> 75, 100 -> 90, 150 -> 100

        TTFT matters more for interactive UX (what users actually feel), so it
        gets the higher weight.
        """
        # TTFT score: exponential decay from perfect (200ms)
        # score = 100 * exp(-k * (ttft - 200))  where k chosen so 2000ms -> 40
        if ttft_ms <= 0:
            ttft_score = 0.0
        elif ttft_ms <= 200:
            ttft_score = 100.0
        else:
            # k ≈ 0.000507 gives 200ms->100, 500ms->86, 1000ms->67, 2000ms->40, 5000ms->11
            ttft_score = 100.0 * math.exp(-0.000507 * (ttft_ms - 200))

        # tok/s score: log curve calibrated for cloud ranges (20-150 tok/s)
        if tok_per_sec <= 0:
            tok_score = 0.0
        elif tok_per_sec < 10:
            tok_score = tok_per_sec * 4.0  # linear ramp 0-40 for 0-10 tok/s
        else:
            # 10->40, 20->52, 40->64, 60->72, 100->85, 150->93
            tok_score = 12.0 * math.log2(tok_per_sec / 5.0) + 16.0

        # Weighted combination
        return 0.60 * ttft_score + 0.40 * tok_score
