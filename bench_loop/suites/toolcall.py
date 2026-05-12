"""ToolCall-15 suite with task-specific tool expectations."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bench_loop.config import TASKS_DIR
from bench_loop.models import BenchmarkTask
from bench_loop.suites.base import BenchmarkSuite


EXPECTATIONS: dict[str, dict[str, Any]] = {
    "tc-01": {"required": [{"name": "get_weather", "args": {"location": "Berlin"}}]},
    "tc-02": {"required": [{"name": "get_stock_price", "args": {"ticker": "AAPL"}}]},
    "tc-03": {"required_any": [{"name": "get_contacts", "args": {"query": "Sarah"}}, {"name": "send_email", "args": {"body": "3pm"}}]},
    "tc-04": {"required": [{"name": "get_weather", "args": {"location": "Tokyo", "units": "fahrenheit"}}]},
    "tc-05": {"required": [{"name": "create_calendar_event", "args": {"title": "standup", "time": "9:30", "duration_minutes": 30}}]},
    "tc-06": {"required": [{"name": "translate_text", "args": {"text": "nearest hospital", "source_language": "English"}}]},
    "tc-07": {"required": [{"name": "search_files", "args": {"query": "Q3 budget"}}]},
    "tc-08": {"required": [{"name": "get_weather", "args": {"location": "Paris"}}]},
    "tc-09": {"required": [{"name": "get_weather", "args": {"location": "London"}}, {"name": "get_stock_price", "args": {"ticker": "MSFT"}}]},
    "tc-10": {"forbid_tools": True, "must_mention": ["1945"]},
    "tc-11": {"forbid_tools": True, "must_mention": ["30"]},
    "tc-12": {"forbid_tools": True, "must_mention_any": ["can't", "cannot", "unable", "delete"]},
    "tc-13": {"required": [{"name": "search_files", "args": {"query": "Johnson proposal"}}]},
    "tc-14": {"required": [{"name": "get_stock_price", "args": {"ticker": "AAPL"}}]},
    "tc-15": {"required": [{"name": "web_search", "args": {"query": "population of Iceland"}}, {"name": "calculator", "args": {"expression": "2%"}}]},
}


class ToolCallSuite(BenchmarkSuite):
    name = "toolcall"
    task_file = Path(TASKS_DIR) / "toolcall" / "tasks.yaml"

    def _normalize(self, value: Any) -> str:
        return str(value).strip().lower()

    def _parse_args(self, value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                return {}
        return {}

    def _extract_calls(self, response: dict[str, Any]) -> list[dict[str, Any]]:
        raw_calls = response.get("tool_calls") or []
        parsed: list[dict[str, Any]] = []
        for call in raw_calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function") if isinstance(call.get("function"), dict) else call
            name = function.get("name")
            args = self._parse_args(function.get("arguments", {}))
            if name:
                parsed.append({"name": name, "args": args})
        raw_response = response.get("raw_response") or {}
        message = (raw_response.get("message") or {}) if isinstance(raw_response, dict) else {}
        for call in (message.get("tool_calls") or []):
            if not isinstance(call, dict):
                continue
            function = call.get("function") if isinstance(call.get("function"), dict) else call
            name = function.get("name")
            args = self._parse_args(function.get("arguments", {}))
            item = {"name": name, "args": args}
            if name and item not in parsed:
                parsed.append(item)
        return parsed

    def _arg_match(self, expected: dict[str, Any], actual: dict[str, Any]) -> bool:
        for key, expected_value in expected.items():
            actual_value = actual.get(key)
            if isinstance(expected_value, (int, float)):
                try:
                    if float(actual_value) != float(expected_value):
                        return False
                except Exception:
                    return False
            else:
                if self._normalize(expected_value) not in self._normalize(actual_value):
                    return False
        return True

    def _call_matches(self, expected_call: dict[str, Any], actual_calls: list[dict[str, Any]]) -> bool:
        for actual in actual_calls:
            if actual.get("name") != expected_call.get("name"):
                continue
            if self._arg_match(expected_call.get("args", {}), actual.get("args", {})):
                return True
        return False

    def evaluate(self, task: BenchmarkTask, response: dict[str, Any]):
        content = self.response_text(response).strip()
        actual_calls = self._extract_calls(response)
        expected = EXPECTATIONS.get(task.id, {})
        notes: list[str] = []
        score = 0.0

        if expected.get("forbid_tools"):
            score = 100.0 if not actual_calls else 0.0
            if actual_calls:
                notes.append("unexpected tool call")
            mentions = expected.get("must_mention") or []
            if mentions and not all(self._normalize(token) in self._normalize(content) for token in mentions):
                score = min(score, 50.0) if score else 0.0
                notes.append("missing expected direct answer content")
            mentions_any = expected.get("must_mention_any") or []
            if mentions_any and not any(self._normalize(token) in self._normalize(content) for token in mentions_any):
                score = min(score, 50.0) if score else 0.0
                notes.append("missing refusal/explanation language")
        else:
            required = expected.get("required", [])
            matched = sum(1 for call in required if self._call_matches(call, actual_calls))
            if required:
                score = 100.0 * matched / len(required)
                if matched < len(required):
                    notes.append(f"matched {matched}/{len(required)} required tool calls")
            required_any = expected.get("required_any", [])
            if required_any:
                any_match = any(self._call_matches(call, actual_calls) for call in required_any)
                score = max(score, 100.0 if any_match else 0.0)
                if not any_match:
                    notes.append("did not match any acceptable tool strategy")

            if not actual_calls and content:
                score = min(score, 25.0) if score else 25.0
                notes.append("answered directly instead of using tools")

        passed = score >= 85.0
        return self.build_result(
            task=task,
            passed=passed,
            score=round(score, 1),
            response=response,
            output=content,
            error="; ".join(notes),
            metadata={
                "actual_tool_calls": actual_calls,
                "evaluation_status": "pass" if passed else ("partial" if score >= 60 else "fail"),
            },
        )
