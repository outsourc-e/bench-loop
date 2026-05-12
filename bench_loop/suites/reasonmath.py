"""ReasonMath-15 suite with answer-line validation."""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from bench_loop.config import TASKS_DIR
from bench_loop.models import BenchmarkTask
from bench_loop.suites.base import BenchmarkSuite


ANSWER_PATTERNS: dict[str, dict[str, Any]] = {
    "rm-01": {"money": 35.98},
    "rm-02": {"number": 0.3125},
    "rm-03": {"pairs": {"new_original_price": 100.0, "saved_money": "yes"}},
    "rm-04": {"contains_any": ["inconsistent", "cannot be determined", "multiple valid", "not uniquely determined"]},
    "rm-05": {"pairs": {"fit": "no", "max_meetings": 3}},
    "rm-06": {"pairs": {"switch": 0.75, "stay": 0.25}},
    "rm-07": {"number": 48.0},
    "rm-08": {"pairs": {"fill_time": 7.2}},
    "rm-09": {"number": 35.0},
    "rm-10": {"money": 2.10},
    "rm-11": {"pairs": {"day": 29}},
    "rm-12": {"pairs": {"person": "son"}},
    "rm-13": {"pairs": {"amount": 5718.96, "interest": 718.96}},
    "rm-14": {"pairs": {"temp_f": 331, "time_min": 34}},
    "rm-15": {"pairs": {"count": 126}},
}


class ReasonMathSuite(BenchmarkSuite):
    name = "reasonmath"
    task_file = Path(TASKS_DIR) / "reasonmath" / "tasks.yaml"

    def _answer_line(self, text: str) -> str:
        for line in reversed([line.strip() for line in text.strip().splitlines()]):
            if line:
                return line
        return ""

    def _extract_number(self, text: str) -> float | None:
        matches = re.findall(r"-?\$?\d+(?:,\d{3})*(?:\.\d+)?", text)
        if not matches:
            return None
        raw = matches[-1].replace("$", "").replace(",", "")
        try:
            return float(raw)
        except ValueError:
            return None

    def _extract_pairs(self, line: str) -> dict[str, str]:
        if ":" in line:
            line = line.split(":", 1)[1].strip()
        parts = [part.strip() for part in line.split(";") if part.strip()]
        pairs: dict[str, str] = {}
        for part in parts:
            if "=" in part:
                key, value = part.split("=", 1)
                pairs[key.strip().lower()] = value.strip()
        return pairs

    def _matches_number(self, actual: float | None, expected: float, tolerance: float = 0.02) -> bool:
        return actual is not None and abs(actual - expected) <= tolerance

    def _score_pairs(self, actual_pairs: dict[str, str], expected_pairs: dict[str, Any]) -> tuple[float, list[str]]:
        notes: list[str] = []
        matched = 0
        for key, expected in expected_pairs.items():
            actual = actual_pairs.get(key.lower())
            if actual is None:
                notes.append(f"missing {key}")
                continue
            if isinstance(expected, str):
                if actual.lower() == expected.lower():
                    matched += 1
                else:
                    notes.append(f"{key} expected {expected}, got {actual}")
            else:
                actual_num = self._extract_number(actual)
                if self._matches_number(actual_num, float(expected)):
                    matched += 1
                else:
                    notes.append(f"{key} expected {expected}, got {actual}")
        score = (matched / max(len(expected_pairs), 1)) * 100.0
        return score, notes

    def evaluate(self, task: BenchmarkTask, response: dict[str, Any]):
        content = self.response_text(response).strip()
        answer_line = self._answer_line(content)
        expected = ANSWER_PATTERNS.get(task.id, {})
        notes: list[str] = []
        score = 0.0

        if not content:
            return self.build_result(
                task=task,
                passed=False,
                score=0.0,
                response=response,
                output="",
                error="Empty response",
                metadata={"evaluation_status": "empty"},
            )

        if not answer_line.startswith("ANSWER:") and task.id != "rm-04":
            notes.append("missing ANSWER line")
            score = 25.0
        if task.id == "rm-04":
            lower = content.lower()
            hits = [phrase for phrase in expected.get("contains_any", []) if phrase in lower]
            score = 100.0 if hits else 0.0
            if not hits:
                notes.append("did not identify inconsistency / non-unique ordering")
        elif "pairs" in expected:
            pairs = self._extract_pairs(answer_line)
            score, notes = self._score_pairs(pairs, expected["pairs"])
        elif "money" in expected:
            actual = self._extract_number(answer_line or content)
            score = 100.0 if self._matches_number(actual, expected["money"]) else 0.0
            if score == 0.0:
                notes.append(f"expected {expected['money']}, got {actual}")
        elif "number" in expected:
            actual = self._extract_number(answer_line or content)
            tolerance = 0.01 if expected["number"] < 1 else 0.05
            score = 100.0 if self._matches_number(actual, expected["number"], tolerance) else 0.0
            if score == 0.0:
                notes.append(f"expected {expected['number']}, got {actual}")

        # small bonus for showing concise work before the answer line
        nonempty_lines = [line.strip() for line in content.splitlines() if line.strip()]
        if score > 0 and len(nonempty_lines) >= 2:
            score = min(100.0, score + 5.0)

        passed = score >= 85.0
        return self.build_result(
            task=task,
            passed=passed,
            score=round(score, 1),
            response=response,
            output=content,
            error="; ".join(notes),
            metadata={
                "answer_line": answer_line,
                "evaluation_status": "pass" if passed else ("partial" if score >= 60 else "fail"),
                "title": task.validation.get("title"),
            },
        )
