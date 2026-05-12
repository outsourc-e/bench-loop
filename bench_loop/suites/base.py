"""Base suite helpers."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import yaml

from bench_loop.models import BenchmarkTask, TaskResult


class BenchmarkSuite:
    """Base class for fixture-backed suites."""

    name: str = "base"
    task_file: Path | None = None

    async def load_tasks(self) -> list[BenchmarkTask]:
        if self.task_file is None:
            return []
        raw = yaml.safe_load(self.task_file.read_text(encoding="utf-8")) or {}
        tasks: list[BenchmarkTask] = []
        for item in raw.get("tasks", []):
            messages = [dict(message) for message in item.get("messages", [])]
            for message in messages:
                content = message.get("content")
                if isinstance(content, str):
                    message["content"] = self.normalize_text(content)
            validation = dict(item.get("validation", {}))
            validation.setdefault("category", item.get("category"))
            validation.setdefault("title", item.get("title"))
            validation.setdefault("scenario_id", str(item.get("id", "")).upper())

            metadata = dict(item.get("metadata", {}))
            capability_tags = list(item.get("capability_tags", metadata.get("capability_tags", [])) or [])
            verifier_type = str(item.get("verifier_type", metadata.get("verifier_type", "")) or "")
            difficulty = str(item.get("difficulty", metadata.get("difficulty", "")) or "")
            expected_turns = item.get("expected_turns", metadata.get("expected_turns"))
            notes = str(item.get("notes", metadata.get("notes", "")) or "")
            if capability_tags:
                validation.setdefault("capability_tags", capability_tags)
            if verifier_type:
                validation.setdefault("verifier_type", verifier_type)
            if difficulty:
                validation.setdefault("difficulty", difficulty)
            if expected_turns is not None:
                validation.setdefault("expected_turns", expected_turns)
            if notes:
                validation.setdefault("notes", notes)

            tasks.append(
                BenchmarkTask(
                    id=item["id"],
                    suite=item.get("suite", self.name),
                    messages=messages,
                    title=str(item.get("title", "") or validation.get("title") or ""),
                    difficulty=difficulty,
                    capability_tags=capability_tags,
                    verifier_type=verifier_type,
                    expected_turns=expected_turns,
                    notes=notes,
                    config=dict(item.get("config", {})),
                    validation=validation,
                    metadata=metadata,
                )
            )
        return tasks

    async def run_task(
        self,
        provider_module: Any,
        endpoint: str,
        model: str,
        task: BenchmarkTask,
        harness: Any | None = None,
        provider_name: str = "ollama",
    ) -> TaskResult:
        request = (
            harness.prepare(task, provider_name=provider_name)
            if harness is not None
            else {"messages": task.messages, **task.config}
        )
        response = await provider_module.chat(
            endpoint=endpoint,
            model=model,
            **request,
        )
        if harness is not None:
            response = harness.postprocess(response, task)
        return self.evaluate(task, response)

    def evaluate(self, task: BenchmarkTask, response: dict[str, Any]) -> TaskResult:
        raise NotImplementedError

    def aggregate_score(self, task_results: list[TaskResult]) -> float:
        return round(sum(task.score for task in task_results) / len(task_results), 2) if task_results else 0.0

    def normalize_text(self, text: str) -> str:
        return (
            text.replace("\\n", "\n")
            .replace("\\t", "\t")
            .replace('\\"', '"')
            .replace("\\'", "'")
            .replace("\\\\", "\\")
        )

    def response_text(self, response: dict[str, Any]) -> str:
        return str(response.get("content") or "")

    def latency_ms(self, response: dict[str, Any]) -> float:
        return float(response.get("total_ms") or 0.0)

    def token_counts(self, response: dict[str, Any]) -> tuple[int, int]:
        return int(response.get("tokens_generated") or 0), int(response.get("tokens_prompt") or 0)

    def build_result(
        self,
        *,
        task: BenchmarkTask,
        passed: bool,
        score: float,
        response: dict[str, Any],
        output: str,
        error: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> TaskResult:
        tokens_generated, tokens_prompt = self.token_counts(response)
        return TaskResult(
            task_id=task.id,
            suite=self.name,
            passed=passed,
            score=score,
            latency_ms=self.latency_ms(response),
            tokens_generated=tokens_generated,
            tokens_prompt=tokens_prompt,
            error=error,
            output=output[:500],
            metadata=metadata or {},
        )

    def now_ms(self) -> float:
        return time.perf_counter() * 1000.0
