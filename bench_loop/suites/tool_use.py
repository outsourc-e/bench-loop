"""Tool use suite fixtures and validation."""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

from bench_loop.config import TASKS_DIR
from bench_loop.models import BenchmarkTask, TaskResult
from bench_loop.suites.base import BenchmarkSuite


class ToolUseSuite(BenchmarkSuite):
    name = "tool_use"
    task_file = Path(TASKS_DIR) / "tool_use" / "tasks.yaml"

    def _parse_json_like(self, text: str) -> dict[str, Any]:
        text = text.strip()
        if not text:
            return {}
        for parser in (json.loads, ast.literal_eval):
            try:
                value = parser(text)
                if isinstance(value, dict):
                    return dict(value)
            except Exception:
                continue
        return {}

    def parse_tool_call(self, response: str) -> tuple[str | None, dict[str, Any]]:
        stripped = response.strip()
        if not stripped:
            return None, {}

        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                if isinstance(parsed.get("function_call"), dict):
                    call = parsed["function_call"]
                    arguments = call.get("arguments", {})
                    if isinstance(arguments, str):
                        arguments = self._parse_json_like(arguments)
                    return call.get("name"), arguments if isinstance(arguments, dict) else {}
                if isinstance(parsed.get("tool_use"), dict):
                    call = parsed["tool_use"]
                    arguments = call.get("input", {})
                    if isinstance(arguments, str):
                        arguments = self._parse_json_like(arguments)
                    return call.get("name"), arguments if isinstance(arguments, dict) else {}
                if "name" in parsed and any(key in parsed for key in ("arguments", "args", "input")):
                    arguments = parsed.get("arguments", parsed.get("args", parsed.get("input", {})))
                    if isinstance(arguments, str):
                        arguments = self._parse_json_like(arguments)
                    return str(parsed.get("name")), arguments if isinstance(arguments, dict) else {}
        except Exception:
            pass

        patterns = [
            r"<tool_use>\s*(\{.*?\})\s*</tool_use>",
            r"<function_call>\s*(\{.*?\})\s*</function_call>",
            r"tool_use\s*:\s*(\{.*\})",
            r"function_call\s*:\s*(\{.*\})",
        ]
        for pattern in patterns:
            match = re.search(pattern, response, flags=re.DOTALL | re.IGNORECASE)
            if match:
                parsed = self._parse_json_like(match.group(1))
                if parsed:
                    if "name" in parsed:
                        arguments = parsed.get("arguments", parsed.get("args", parsed.get("input", {})))
                        if isinstance(arguments, str):
                            arguments = self._parse_json_like(arguments)
                        return str(parsed["name"]), arguments if isinstance(arguments, dict) else {}

        direct_call_match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\((.*)\)", stripped, flags=re.DOTALL)
        if direct_call_match:
            tool_name = direct_call_match.group(1)
            args_text = direct_call_match.group(2).strip()
            kwargs: dict[str, Any] = {}
            if args_text:
                try:
                    fake_call = f"f({args_text})"
                    expr = ast.parse(fake_call, mode="eval")
                    if isinstance(expr.body, ast.Call):
                        for keyword in expr.body.keywords:
                            kwargs[keyword.arg or ""] = ast.literal_eval(keyword.value)
                except Exception:
                    pass
            return tool_name, kwargs

        return None, {}

    def _extract_tool_call(self, response: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
        tool_calls = response.get("tool_calls") or []
        if isinstance(tool_calls, list) and tool_calls:
            first_call = tool_calls[0] or {}
            function = first_call.get("function") if isinstance(first_call, dict) else None
            if isinstance(function, dict):
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    arguments = self._parse_json_like(arguments)
                return function.get("name"), arguments if isinstance(arguments, dict) else {}
        response_text = str(response.get("content") or "")
        return self.parse_tool_call(response_text)

    def _matches_subset(self, expected: dict[str, Any], actual: dict[str, Any]) -> bool:
        for key, expected_value in expected.items():
            actual_value = actual.get(key)
            if isinstance(expected_value, str) and isinstance(actual_value, str):
                if expected_value.lower() not in actual_value.lower():
                    return False
            elif isinstance(expected_value, list) and isinstance(actual_value, list):
                for item in expected_value:
                    if item not in actual_value:
                        return False
            elif actual_value != expected_value:
                return False
        return True

    def evaluate(self, task: BenchmarkTask, response: dict[str, Any]) -> TaskResult:
        response_text = self.response_text(response)
        expected_tool = task.validation.get("expected_tool")
        expected_args = dict(task.validation.get("expected_args", {}))
        tool_name, tool_args = self._extract_tool_call(response)

        if expected_tool is None:
            passed = tool_name is None
            error = "Unexpected tool call" if tool_name is not None else ""
        else:
            passed = tool_name == expected_tool and self._matches_subset(expected_args, tool_args)
            if tool_name != expected_tool:
                error = f"Expected tool {expected_tool}, got {tool_name}"
            elif not self._matches_subset(expected_args, tool_args):
                error = f"Expected args subset {expected_args}, got {tool_args}"
            else:
                error = ""

        return self.build_result(
            task=task,
            passed=passed,
            score=100.0 if passed else 0.0,
            response=response,
            output=response_text,
            error=error,
            metadata={
                "tool_name": tool_name,
                "tool_args": tool_args,
                "tool_calls": response.get("tool_calls") or [],
            },
        )
