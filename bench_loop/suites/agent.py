"""Agent suite — multi-turn benchmarks where BenchLoop runs the harness loop.

This suite is fundamentally different from the single-shot suites. Each task
defines a goal and a small set of *real* tools BenchLoop will execute on the
model's behalf. The model is expected to:

  1. Plan
  2. Call a tool
  3. Observe the tool result (fed back as a `tool` message)
  4. Iterate until it can answer the user's question
  5. Emit a final assistant message containing the answer

BenchLoop drives the conversation turn-by-turn, executes each tool call
locally (sandboxed: pure-Python calculator, in-memory file system, frozen
fake-internet table), and feeds results back to the model through whichever
harness adapter the run is using. That makes this the first suite where
"harness" actually means something operational — it controls the multi-turn
prompt grammar, not just a single-shot wrapper.

Scoring (per task):
  - correct_final: model's final answer matches expected substring(s)
  - efficient: completed in <= max_turns (configurable, defaults 6)
  - no_hallucinated_tools: every tool call used a defined tool name
  - all_required_called: every `must_call` tool from the spec was invoked at
    least once with the expected args
Each criterion is worth 25 pts; total 100 per task. Average across tasks =
suite score.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bench_loop.config import TASKS_DIR
from bench_loop.models import BenchmarkTask, TaskResult
from bench_loop.suites.base import BenchmarkSuite


# --------------------------------------------------------------------------- #
# Built-in tool implementations                                                #
#                                                                              #
# Each tool is a pure-Python callable that takes a dict of args and returns a  #
# string (the "tool result" the model will see on the next turn). These are    #
# deterministic on purpose — repeated benchmark runs must produce identical   #
# observations.                                                                #
# --------------------------------------------------------------------------- #
def _tool_calculator(args: dict[str, Any]) -> str:
    expr = str(args.get("expression", "")).strip()
    if not expr:
        return "ERROR: missing 'expression' argument"
    # Whitelist characters to avoid eval shenanigans.
    if not re.fullmatch(r"[\d\s+\-*/().%,eE]+", expr):
        return f"ERROR: expression contains forbidden characters: {expr!r}"
    try:
        # eval is safe here because of the whitelist above.
        value = eval(expr, {"__builtins__": {}}, {})  # noqa: S307
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"
    return str(value)


_WEATHER = {
    "san francisco": {"temp_f": 62, "condition": "fog"},
    "berlin": {"temp_f": 48, "condition": "rain"},
    "tokyo": {"temp_f": 71, "condition": "clear"},
    "paris": {"temp_f": 55, "condition": "overcast"},
    "london": {"temp_f": 52, "condition": "drizzle"},
    "nyc": {"temp_f": 58, "condition": "partly cloudy"},
    "new york": {"temp_f": 58, "condition": "partly cloudy"},
}


def _tool_weather(args: dict[str, Any]) -> str:
    loc = str(args.get("location", "")).strip().lower()
    if loc not in _WEATHER:
        return f"ERROR: no data for {loc!r}. Try one of: {', '.join(sorted(_WEATHER))}"
    data = _WEATHER[loc]
    units = str(args.get("units", "fahrenheit")).strip().lower()
    if units.startswith("c"):
        temp_c = round((data["temp_f"] - 32) * 5 / 9, 1)
        return f"{loc.title()}: {temp_c}°C, {data['condition']}"
    return f"{loc.title()}: {data['temp_f']}°F, {data['condition']}"


_STOCKS = {
    "AAPL": 188.43,
    "MSFT": 412.65,
    "GOOGL": 142.07,
    "TSLA": 198.50,
    "NVDA": 925.10,
}


def _tool_stock(args: dict[str, Any]) -> str:
    ticker = str(args.get("ticker", "")).strip().upper()
    if ticker not in _STOCKS:
        return f"ERROR: unknown ticker {ticker!r}. Try: {', '.join(_STOCKS)}"
    return f"{ticker}: ${_STOCKS[ticker]:.2f}"


def _tool_word_count(args: dict[str, Any]) -> str:
    text = str(args.get("text", ""))
    if not text:
        return "ERROR: missing 'text' argument"
    return str(len(text.split()))


def _tool_reverse(args: dict[str, Any]) -> str:
    text = str(args.get("text", ""))
    return text[::-1]


# Tool spec passed to the model (OpenAI-style schema)
TOOL_SCHEMAS = {
    "calculator": {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression. Supports +, -, *, /, %, parentheses.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "Math expression to evaluate."}},
                "required": ["expression"],
            },
        },
    },
    "get_weather": {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city. Returns temperature and condition.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name."},
                    "units": {"type": "string", "enum": ["fahrenheit", "celsius"], "default": "fahrenheit"},
                },
                "required": ["location"],
            },
        },
    },
    "get_stock_price": {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the latest stock price for a ticker symbol.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string", "description": "Ticker symbol, e.g. AAPL."}},
                "required": ["ticker"],
            },
        },
    },
    "word_count": {
        "type": "function",
        "function": {
            "name": "word_count",
            "description": "Count words in a string.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    },
    "reverse_text": {
        "type": "function",
        "function": {
            "name": "reverse_text",
            "description": "Reverse a string character-by-character.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    },
}

TOOL_IMPL = {
    "calculator": _tool_calculator,
    "get_weather": _tool_weather,
    "get_stock_price": _tool_stock,
    "word_count": _tool_word_count,
    "reverse_text": _tool_reverse,
}


# --------------------------------------------------------------------------- #
# Agent loop                                                                   #
# --------------------------------------------------------------------------- #
@dataclass
class AgentTurn:
    role: str            # "user" | "assistant" | "tool"
    content: str
    tool_calls: list[dict[str, Any]] | None = None
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_result: str | None = None


@dataclass
class AgentTrace:
    turns: list[AgentTurn]
    final_answer: str
    completed: bool
    stop_reason: str
    tool_calls_total: int
    hallucinated_tools: list[str]
    required_satisfied: bool


class AgentSuite(BenchmarkSuite):
    name = "agent"
    task_file = Path(TASKS_DIR) / "agent" / "tasks.yaml"

    # Default max turns. Tasks can override via `max_turns` in the yaml.
    DEFAULT_MAX_TURNS = 6

    def evaluate(self, task: BenchmarkTask, response: dict[str, Any]) -> TaskResult:
        """Evaluate a single agent task. `response` here is the full agent trace,
        not a single chat response — populated by `run_task` below.
        """
        trace: AgentTrace = response["__trace"]
        validation = task.validation or {}
        expected_contains = validation.get("expected_contains", []) or []
        if isinstance(expected_contains, str):
            expected_contains = [expected_contains]

        max_turns = int(validation.get("max_turns", self.DEFAULT_MAX_TURNS))

        # Criteria — each worth 25 pts.
        final_lower = (trace.final_answer or "").lower()
        correct_final = bool(expected_contains) and all(
            str(needle).lower() in final_lower for needle in expected_contains
        )
        efficient = trace.completed and len(trace.turns) <= max_turns * 2  # user+assistant per turn
        no_hallucination = len(trace.hallucinated_tools) == 0
        all_required_called = trace.required_satisfied

        components = {
            "correct_final": 25 if correct_final else 0,
            "efficient": 25 if efficient else 0,
            "no_hallucinated_tools": 25 if no_hallucination else 0,
            "all_required_called": 25 if all_required_called else 0,
        }
        score = float(sum(components.values()))
        passed = correct_final and no_hallucination and all_required_called

        return self.build_result(
            task=task,
            passed=passed,
            score=score,
            response={"content": trace.final_answer},
            output=trace.final_answer[:500],
            metadata={
                "agent_components": components,
                "turns": [
                    {
                        "role": t.role,
                        "content": t.content[:400],
                        "tool_name": t.tool_name,
                        "tool_args": t.tool_args,
                        "tool_result": (t.tool_result or "")[:400],
                    }
                    for t in trace.turns
                ],
                "tool_calls_total": trace.tool_calls_total,
                "hallucinated_tools": trace.hallucinated_tools,
                "completed": trace.completed,
                "stop_reason": trace.stop_reason,
                "max_turns": max_turns,
            },
        )

    async def run_task(
        self,
        provider_module: Any,
        endpoint: str,
        model: str,
        task: BenchmarkTask,
        harness: Any | None = None,
        provider_name: str = "ollama",
    ) -> TaskResult:
        """Run a multi-turn agent conversation against the model, executing tools
        between turns. This OVERRIDES the default single-shot run_task and is
        the heart of the agent suite.
        """
        validation = task.validation or {}
        max_turns = int(validation.get("max_turns", self.DEFAULT_MAX_TURNS))
        allowed_tools = list(validation.get("tools", list(TOOL_SCHEMAS)))
        required_calls = list(validation.get("must_call", []))  # [{"name": "...", "args_contains": {...}}]

        # Build the tool definitions the model sees this run.
        tool_schemas = [TOOL_SCHEMAS[name] for name in allowed_tools if name in TOOL_SCHEMAS]

        # Start the conversation with the task's first user turn.
        messages = [dict(m) for m in task.messages]
        turns: list[AgentTurn] = []
        for m in messages:
            turns.append(AgentTurn(role=m.get("role", "user"), content=str(m.get("content", ""))))

        hallucinated: list[str] = []
        tool_calls_total = 0
        required_seen: set[str] = set()
        stop_reason = "max_turns"
        final_answer = ""
        completed = False

        for turn_idx in range(max_turns):
            # Prepare a task-like wrapper so the harness can inject its
            # contract. We synthesize a one-shot BenchmarkTask carrying the
            # current message list + tool schemas.
            synthetic_task = BenchmarkTask(
                id=task.id,
                suite=self.name,
                messages=messages,
                config={**task.config, "tools": tool_schemas, "max_tokens": 512},
                validation=validation,
                metadata=task.metadata,
            )
            request = (
                harness.prepare(synthetic_task, provider_name=provider_name)
                if harness is not None
                else {"messages": messages, "tools": tool_schemas, "max_tokens": 512}
            )

            response = await provider_module.chat(
                endpoint=endpoint,
                model=model,
                **request,
            )
            if harness is not None:
                response = harness.postprocess(response, synthetic_task)

            content = (response.get("content") or "").strip()
            tool_calls = response.get("tool_calls") or []

            # Record the assistant turn
            assistant_turn = AgentTurn(
                role="assistant",
                content=content,
                tool_calls=tool_calls or None,
            )
            turns.append(assistant_turn)

            # If the model didn't call a tool, treat its message as the final answer.
            if not tool_calls:
                final_answer = content
                completed = True
                stop_reason = "model_finished"
                break

            # Execute each tool call in order, append the result as a tool message.
            # Important: format depends on provider.
            #   - OpenAI-compatible: tool_call has id + type, arguments is a JSON string,
            #     and the follow-up `tool` message uses `tool_call_id`.
            #   - Ollama: tool_calls have no id, arguments must be an object, and the
            #     follow-up `tool` message must NOT include `tool_call_id`.
            is_ollama = provider_name == "ollama"

            def _serialize_args(arg_val: Any) -> Any:
                # Ollama wants a dict object; OpenAI-compat wants a JSON string.
                if is_ollama:
                    if isinstance(arg_val, dict):
                        return arg_val
                    if isinstance(arg_val, str):
                        try:
                            parsed = json.loads(arg_val) if arg_val else {}
                            return parsed if isinstance(parsed, dict) else {}
                        except Exception:
                            return {}
                    return {}
                else:
                    if isinstance(arg_val, str):
                        return arg_val
                    return json.dumps(arg_val or {}, ensure_ascii=False)

            assistant_message_for_history: dict[str, Any] = {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    (
                        {
                            "function": {
                                "name": (call.get("function") or {}).get("name") if isinstance(call, dict) else "",
                                "arguments": _serialize_args(
                                    (call.get("function") or {}).get("arguments", {}) if isinstance(call, dict) else {}
                                ),
                            },
                        }
                        if is_ollama
                        else {
                            "id": f"call_{turn_idx}_{i}",
                            "type": "function",
                            "function": {
                                "name": (call.get("function") or {}).get("name") if isinstance(call, dict) else "",
                                "arguments": _serialize_args(
                                    (call.get("function") or {}).get("arguments", "{}") if isinstance(call, dict) else "{}"
                                ),
                            },
                        }
                    )
                    for i, call in enumerate(tool_calls)
                ],
            }
            messages.append(assistant_message_for_history)

            for i, call in enumerate(tool_calls):
                fn = call.get("function") if isinstance(call, dict) else {}
                name = (fn or {}).get("name") or ""
                args_raw = (fn or {}).get("arguments") or "{}"
                if isinstance(args_raw, dict):
                    args = args_raw
                else:
                    try:
                        args = json.loads(args_raw)
                    except Exception:
                        args = {}

                tool_calls_total += 1

                if name not in TOOL_IMPL:
                    hallucinated.append(name)
                    tool_result = f"ERROR: unknown tool {name!r}. Available: {', '.join(allowed_tools)}"
                else:
                    try:
                        tool_result = TOOL_IMPL[name](args)
                    except Exception as exc:
                        tool_result = f"ERROR: tool execution failed: {type(exc).__name__}: {exc}"

                # Track required calls
                for req in required_calls:
                    if req.get("name") == name:
                        wanted_args = req.get("args_contains", {})
                        if all(str(args.get(k, "")).lower().find(str(v).lower()) >= 0 for k, v in wanted_args.items()):
                            required_seen.add(name)

                turns.append(AgentTurn(
                    role="tool",
                    content=tool_result,
                    tool_name=name,
                    tool_args=args,
                    tool_result=tool_result,
                ))
                tool_message: dict[str, Any] = {
                    "role": "tool",
                    "name": name,
                    "content": tool_result,
                }
                if not is_ollama:
                    tool_message["tool_call_id"] = f"call_{turn_idx}_{i}"
                messages.append(tool_message)

        else:
            # for/else: ran out of turns
            stop_reason = "max_turns_exceeded"

        required_satisfied = (
            len(required_calls) == 0 or {r.get("name") for r in required_calls}.issubset(required_seen)
        )

        trace = AgentTrace(
            turns=turns,
            final_answer=final_answer,
            completed=completed,
            stop_reason=stop_reason,
            tool_calls_total=tool_calls_total,
            hallucinated_tools=hallucinated,
            required_satisfied=required_satisfied,
        )

        return self.evaluate(task, {"__trace": trace})
