"""DataExtract-15 suite."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bench_loop.config import TASKS_DIR
from bench_loop.models import BenchmarkTask, TaskResult
from bench_loop.suites.base import BenchmarkSuite


CATEGORY_WEIGHTS = {"A": 15, "B": 20, "C": 25, "D": 25, "E": 15}
ARRAY_OBJECT_ANCHORS = {
    "DE-02.items": "name",
    "DE-07.$root": "name",
    "DE-13.line_items": "description",
    "DE-13.discounts": "description",
}


class DataExtractSuite(BenchmarkSuite):
    name = "dataextract"
    task_file = Path(TASKS_DIR) / "dataextract" / "tasks.yaml"

    def _status_for_score(self, score: int) -> str:
        if score >= 85:
            return "pass"
        if score >= 60:
            return "partial"
        return "fail"

    def _is_plain_object(self, value: Any) -> bool:
        return isinstance(value, dict)

    def _top_level_shape(self, value: Any) -> str:
        if isinstance(value, list):
            return "array"
        if isinstance(value, dict):
            return "object"
        return "other"

    def _normalize_string(self, value: str) -> str:
        return value.strip()

    def _compare_scalar(self, expected: Any, actual: Any) -> tuple[bool, str | None]:
        if expected is None:
            return actual is None, None if actual is None else "expected null"
        if isinstance(expected, str):
            return isinstance(actual, str) and self._normalize_string(actual) == self._normalize_string(expected), (
                None if isinstance(actual, str) else "expected string"
            )
        if isinstance(expected, bool):
            return actual is expected, None if isinstance(actual, bool) else "expected boolean"
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            return (
                isinstance(actual, (int, float))
                and not isinstance(actual, bool)
                and abs(float(actual) - float(expected)) <= 0.01,
                None if isinstance(actual, (int, float)) and not isinstance(actual, bool) else "expected number",
            )
        return False, "unsupported scalar type"

    def _compare_scalar_array(self, expected: list[Any], actual: Any) -> tuple[int, int, list[str]]:
        if not isinstance(actual, list):
            return 0, 1, ["expected array"]
        if len(expected) != len(actual):
            return 0, 1, [f"expected {len(expected)} items but received {len(actual)}"]
        remaining = list(actual)
        for expected_item in expected:
            match_index = -1
            for index, candidate in enumerate(remaining):
                ok, _ = self._compare_scalar(expected_item, candidate)
                if ok:
                    match_index = index
                    break
            if match_index == -1:
                return 0, 1, ["array values did not match expected set"]
            remaining.pop(match_index)
        return 1, 1, []

    def _compare_object_array(
        self,
        expected: list[dict[str, Any]],
        actual: Any,
        scenario_id: str,
        path: str,
    ) -> tuple[int, int, list[str]]:
        if not isinstance(actual, list):
            width = len(expected[0].keys()) if expected and expected[0] else 1
            return 0, len(expected) * width, ["expected array"]
        anchor = ARRAY_OBJECT_ANCHORS.get(f"{scenario_id}.{path or '$root'}")
        if not anchor:
            return 0, len(expected) or 1, [f"missing anchor key for {scenario_id}.{path or '$root'}"]
        actual_by_anchor: dict[str, dict[str, Any]] = {}
        for item in actual:
            if isinstance(item, dict) and isinstance(item.get(anchor), str):
                actual_by_anchor[str(item[anchor])] = item
        correct = 0
        total = 0
        notes: list[str] = []
        for expected_item in expected:
            actual_item = actual_by_anchor.get(str(expected_item[anchor]))
            for key, expected_value in expected_item.items():
                total += 1
                if actual_item is None:
                    notes.append(f"missing object with {anchor}={expected_item[anchor]}")
                    continue
                sub_correct, sub_total, sub_notes = self._compare_value(
                    expected_value,
                    actual_item.get(key),
                    scenario_id,
                    f"{path}.{key}" if path else key,
                )
                correct += sub_correct
                notes.extend(sub_notes)
        return correct, total, notes

    def _compare_object(
        self,
        expected: dict[str, Any],
        actual: Any,
        scenario_id: str,
        path: str = "",
    ) -> tuple[int, int, list[str]]:
        if not isinstance(actual, dict):
            return 0, len(expected), ["expected object"]
        correct = 0
        total = 0
        notes: list[str] = []
        for key, expected_value in expected.items():
            nested_path = f"{path}.{key}" if path else key
            sub_correct, sub_total, sub_notes = self._compare_value(expected_value, actual.get(key), scenario_id, nested_path)
            correct += sub_correct
            total += sub_total
            notes.extend(sub_notes)
        return correct, total, notes

    def _compare_value(self, expected: Any, actual: Any, scenario_id: str, path: str) -> tuple[int, int, list[str]]:
        if isinstance(expected, list):
            if all(isinstance(item, dict) for item in expected):
                return self._compare_object_array(expected, actual, scenario_id, path)
            return self._compare_scalar_array(expected, actual)
        if isinstance(expected, dict):
            return self._compare_object(expected, actual, scenario_id, path)
        ok, reason = self._compare_scalar(expected, actual)
        return (1 if ok else 0), 1, ([] if ok else [f"{path}: {reason or 'mismatch'}"])

    def _evaluate_compliance(self, expected: Any, actual: Any) -> tuple[bool, bool, bool, list[str]]:
        notes: list[str] = []
        exact_top_level_shape = self._top_level_shape(expected) == self._top_level_shape(actual)
        if not exact_top_level_shape:
            notes.append(
                f"top-level shape mismatch: expected {self._top_level_shape(expected)}, received {self._top_level_shape(actual)}"
            )
        requested_fields_only = True
        no_missing_expected_fields = True
        if isinstance(expected, dict) and isinstance(actual, dict):
            expected_keys = set(expected.keys())
            actual_keys = set(actual.keys())
            extra = sorted(actual_keys - expected_keys)
            missing = sorted(expected_keys - actual_keys)
            if extra:
                requested_fields_only = False
                notes.append(f"extra top-level fields: {', '.join(extra)}")
            if missing:
                no_missing_expected_fields = False
                notes.append(f"missing top-level fields: {', '.join(missing)}")
        return exact_top_level_shape, requested_fields_only, no_missing_expected_fields, notes

    def evaluate(self, task: BenchmarkTask, response: dict[str, Any]) -> TaskResult:
        response_text = self.response_text(response)
        expected = task.validation.get("expected")
        scenario_id = str(task.validation.get("scenario_id") or task.id.upper())
        try:
            parsed = json.loads(response_text)
        except Exception as exc:
            return self.build_result(
                task=task,
                passed=False,
                score=0.0,
                response=response,
                output=response_text,
                error=f"Invalid JSON: {exc}",
                metadata={
                    "scenario_id": scenario_id,
                    "evaluation_status": "invalid_json",
                    "summary": f"Invalid JSON: {exc}",
                    "note": "Official score is 0 when the response is not valid JSON.",
                    "category": task.validation.get("category"),
                    "title": task.validation.get("title"),
                },
            )
        exact_shape, fields_only, no_missing, compliance_notes = self._evaluate_compliance(expected, parsed)
        correct, total, notes = self._compare_value(expected, parsed, scenario_id, "")
        score = 0 if total == 0 else round(correct / total * 100)
        summary = (
            f"{correct}/{total} atomic fields correct ({score}%). "
            f"{'shape ok' if exact_shape else 'shape fail'}, "
            f"{'fields only' if fields_only else 'extra fields'}, "
            f"{'no missing fields' if no_missing else 'missing fields'}."
        )
        note = " | ".join(compliance_notes + notes) if (compliance_notes or notes) else ""
        passed = score >= 85
        return self.build_result(
            task=task,
            passed=passed,
            score=float(score),
            response=response,
            output=response_text,
            error="" if passed else note,
            metadata={
                "scenario_id": scenario_id,
                "evaluation_status": self._status_for_score(score),
                "summary": summary,
                "note": note,
                "category": task.validation.get("category"),
                "title": task.validation.get("title"),
            },
        )

    def aggregate_score(self, task_results: list[TaskResult]) -> float:
        grouped: dict[str, list[float]] = {key: [] for key in CATEGORY_WEIGHTS}
        for result in task_results:
            category = str(result.metadata.get("category") or "")
            if category in grouped:
                grouped[category].append(result.score)
        weighted = 0.0
        for category, weight in CATEGORY_WEIGHTS.items():
            avg = sum(grouped[category]) / len(grouped[category]) if grouped[category] else 0.0
            weighted += avg * (weight / 100.0)
        return round(weighted, 2)
