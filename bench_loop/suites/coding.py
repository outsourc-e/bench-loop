"""Coding suite execution and evaluation."""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from bench_loop.config import TASKS_DIR
from bench_loop.models import BenchmarkTask, TaskResult
from bench_loop.suites.base import BenchmarkSuite


CODE_BLOCK_RE = re.compile(r"```python\s*(.*?)```", re.DOTALL | re.IGNORECASE)
GENERIC_BLOCK_RE = re.compile(r"```\s*(.*?)```", re.DOTALL)


class CodingSuite(BenchmarkSuite):
    name = "coding"
    task_file = Path(TASKS_DIR) / "coding" / "tasks.yaml"

    def _extract_code(self, response_text: str) -> str:
        match = CODE_BLOCK_RE.search(response_text)
        if match:
            return match.group(1).strip()
        match = GENERIC_BLOCK_RE.search(response_text)
        if match:
            return match.group(1).strip()
        return response_text.strip()

    def evaluate(self, task: BenchmarkTask, response: dict[str, Any]) -> TaskResult:
        response_text = str(response.get("content") or "")
        code = self._extract_code(response_text)
        test_code = str(task.validation.get("test_code") or "")
        if not code:
            return self.build_result(
                task=task,
                passed=False,
                score=0.0,
                response=response,
                output=response_text,
                error="No code found in model response",
                metadata={"evaluation_status": "missing_code"},
            )

        try:
            compile(code, f"<{task.id}>", "exec")
        except SyntaxError as exc:
            return self.build_result(
                task=task,
                passed=False,
                score=0.0,
                response=response,
                output=code,
                error=f"SyntaxError: {exc.msg} (line {exc.lineno})",
                metadata={"evaluation_status": "syntax_error"},
            )

        script = f"{code}\n\n{test_code}\n"
        with tempfile.TemporaryDirectory(prefix="bench-loop-coding-") as temp_dir:
            script_path = Path(temp_dir) / "task.py"
            script_path.write_text(script, encoding="utf-8")
            try:
                completed = subprocess.run(
                    [sys.executable, str(script_path)],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                stdout = completed.stdout or ""
                stderr = completed.stderr or ""
            except subprocess.TimeoutExpired:
                stdout = ""
                stderr = "Timed out after 10s"
                completed = None

        if completed is not None and completed.returncode == 0 and "PASS" in stdout:
            passed = True
            score = 100.0
            error = ""
            status = "all_tests_passed"
        else:
            passed = False
            score = 25.0
            status = "tests_failed_or_runtime_error"
            combined_error = (stderr.strip() or stdout.strip() or "")
            if completed is None:
                error = stderr
            elif "SyntaxError" in combined_error:
                score = 0.0
                status = "syntax_error"
                error = combined_error
            elif completed.returncode != 0:
                error = combined_error or f"exit code {completed.returncode}"
            else:
                error = combined_error or "Tests did not report PASS"

        return self.build_result(
            task=task,
            passed=passed,
            score=score,
            response=response,
            output=code,
            error=error,
            metadata={
                "stdout": stdout,
                "stderr": stderr,
                "evaluation_status": status,
            },
        )
