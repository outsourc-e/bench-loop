"""Harness adapters for BenchLoop.

A harness wraps a benchmark task on the way out (system prompt, tool plumbing)
and parses model output on the way back (extracting tool calls, reasoning, etc.).
This lets us compare how the SAME model performs under different prompting /
parsing contracts (raw OpenAI tools, Hermes tags, Qwen3 function-call tags).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


# --------------------------------------------------------------------------- #
# Raw passthrough — what the engine has always done                            #
# --------------------------------------------------------------------------- #
@dataclass
class RawHarness:
    name: str = "raw"
    version: str = "v1"

    def prepare(self, task: Any, provider_name: str = "ollama") -> dict[str, Any]:
        return {"messages": task.messages, **task.config}

    def postprocess(self, response: dict[str, Any], task: Any) -> dict[str, Any]:
        return response


# --------------------------------------------------------------------------- #
# Hermes — NousResearch's tool-call format                                     #
#                                                                              #
# System prompt teaches the model to emit                                      #
#   <tool_call>{"name": "fn", "arguments": {...}}</tool_call>                  #
# (one block per call). We strip the OpenAI-style `tools=` param and inline    #
# the JSON-schema tool definitions inside the system prompt; postprocess pulls #
# every <tool_call> block out of the text and re-injects them as              #
# response.tool_calls so the existing evaluator works unchanged.               #
# --------------------------------------------------------------------------- #
HERMES_SYSTEM_TEMPLATE = """You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. Here are the available tools:

<tools>
{tool_block}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
<tool_call>
{{"name": "<function-name>", "arguments": <args-json-object>}}
</tool_call>

If a tool is not necessary to answer, reply directly without any <tool_call> block.
"""

_HERMES_BLOCK_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL | re.IGNORECASE)


def _format_tool_block(tools: list[dict[str, Any]]) -> str:
    """Render OpenAI-style tools list as Hermes <tools> JSON-schema lines."""
    if not tools:
        return ""
    lines: list[str] = []
    for tool in tools:
        fn = tool.get("function") if isinstance(tool, dict) else None
        if not fn:
            continue
        lines.append(json.dumps(fn, ensure_ascii=False))
    return "\n".join(lines)


@dataclass
class HermesHarness:
    name: str = "hermes"
    version: str = "v1"

    def prepare(self, task: Any, provider_name: str = "ollama") -> dict[str, Any]:
        messages = [dict(msg) for msg in task.messages]
        config = dict(task.config)
        tools = config.pop("tools", []) or []

        if tools:
            tool_block = _format_tool_block(tools)
            hermes_sys = HERMES_SYSTEM_TEMPLATE.format(tool_block=tool_block)
            # Prepend Hermes system prompt; preserve any existing system content as a second system message.
            extra_sys = [m for m in messages if m.get("role") == "system"]
            non_sys = [m for m in messages if m.get("role") != "system"]
            merged_sys = hermes_sys
            for m in extra_sys:
                merged_sys += "\n\n" + (m.get("content") or "")
            messages = [{"role": "system", "content": merged_sys}, *non_sys]

        return {"messages": messages, **config}

    def postprocess(self, response: dict[str, Any], task: Any) -> dict[str, Any]:
        return _parse_tagged_calls(response, _HERMES_BLOCK_RE, source="hermes_tool_call")


# --------------------------------------------------------------------------- #
# Qwen — Qwen2/Qwen3 family <function_call> tags                              #
# Qwen3-Coder and Qwen-Agent use a similar XML wrapper but different tag.     #
# --------------------------------------------------------------------------- #
QWEN_SYSTEM_TEMPLATE = """You are an AI assistant with access to the following tools. Call a tool by emitting a single XML block:
<function_call>
{{"name": "<function-name>", "arguments": <args-json-object>}}
</function_call>

Available tools:
{tool_block}

Only call a tool when it is necessary. If you can answer directly, reply in plain text.
"""

_QWEN_BLOCK_RE = re.compile(r"<function_call>\s*(\{.*?\})\s*</function_call>", re.DOTALL | re.IGNORECASE)


@dataclass
class QwenHarness:
    name: str = "qwen"
    version: str = "v1"

    def prepare(self, task: Any, provider_name: str = "ollama") -> dict[str, Any]:
        messages = [dict(msg) for msg in task.messages]
        config = dict(task.config)
        tools = config.pop("tools", []) or []
        if tools:
            tool_block = _format_tool_block(tools)
            qwen_sys = QWEN_SYSTEM_TEMPLATE.format(tool_block=tool_block)
            extra_sys = [m for m in messages if m.get("role") == "system"]
            non_sys = [m for m in messages if m.get("role") != "system"]
            merged = qwen_sys + ("\n\n" + "\n".join(m.get("content", "") for m in extra_sys) if extra_sys else "")
            messages = [{"role": "system", "content": merged}, *non_sys]
        return {"messages": messages, **config}

    def postprocess(self, response: dict[str, Any], task: Any) -> dict[str, Any]:
        return _parse_tagged_calls(response, _QWEN_BLOCK_RE, source="qwen_function_call")


# --------------------------------------------------------------------------- #
# OCPlatform / Pi / generic <think>...</think> + tool-call thinking models      #
# Splits reasoning from final answer so downstream eval doesn't penalize       #
# verbose chain-of-thought. Tool calls still use Hermes <tool_call> tags.      #
# --------------------------------------------------------------------------- #
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)

PI_SYSTEM_TEMPLATE = """You are an AI assistant. First reason inside a <think>...</think> block, then give your final answer outside the block. {tool_clause}"""

PI_TOOL_CLAUSE = """When you need to call a tool, emit it in a <tool_call>...</tool_call> block containing JSON like {{"name": "fn", "arguments": {{...}}}}.

Available tools:
{tool_block}"""


@dataclass
class PiHarness:
    """OCPlatform / Pi-style — reasoning trace in <think> tags + Hermes tool format."""
    name: str = "pi"
    version: str = "v1"

    def prepare(self, task: Any, provider_name: str = "ollama") -> dict[str, Any]:
        messages = [dict(msg) for msg in task.messages]
        config = dict(task.config)
        tools = config.pop("tools", []) or []
        tool_clause = ""
        if tools:
            tool_clause = PI_TOOL_CLAUSE.format(tool_block=_format_tool_block(tools))
        pi_sys = PI_SYSTEM_TEMPLATE.format(tool_clause=tool_clause).strip()
        extra_sys = [m for m in messages if m.get("role") == "system"]
        non_sys = [m for m in messages if m.get("role") != "system"]
        merged = pi_sys + ("\n\n" + "\n".join(m.get("content", "") for m in extra_sys) if extra_sys else "")
        messages = [{"role": "system", "content": merged}, *non_sys]
        return {"messages": messages, **config}

    def postprocess(self, response: dict[str, Any], task: Any) -> dict[str, Any]:
        if not isinstance(response, dict):
            return response
        content = response.get("content") or ""
        if not isinstance(content, str):
            return response

        # Extract think blocks into metadata, strip from content so scorer sees only the answer.
        thinks = _THINK_RE.findall(content)
        stripped = _THINK_RE.sub("", content).strip()
        response = {**response, "content": stripped}
        if thinks:
            response.setdefault("metadata", {})
            if isinstance(response["metadata"], dict):
                response["metadata"] = {
                    **response["metadata"],
                    "reasoning_blocks": len(thinks),
                    "reasoning_chars": sum(len(t) for t in thinks),
                }

        # Then parse Hermes-style tool calls.
        return _parse_tagged_calls(response, _HERMES_BLOCK_RE, source="pi_tool_call")


# --------------------------------------------------------------------------- #
# Shared tag parser — keeps Hermes/Qwen/Pi logic DRY                          #
# --------------------------------------------------------------------------- #
def _parse_tagged_calls(response: dict[str, Any], pattern: re.Pattern, source: str = "") -> dict[str, Any]:
    if not isinstance(response, dict):
        return response
    content = response.get("content") or ""
    if not isinstance(content, str):
        return response

    existing_calls = list(response.get("tool_calls") or [])
    parsed_calls: list[dict[str, Any]] = []
    stripped_content = content
    for match in pattern.finditer(content):
        blob = match.group(1).strip()
        try:
            obj = json.loads(blob)
        except Exception:
            try:
                obj = json.loads(blob.replace("\n", " "))
            except Exception:
                continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("name") or obj.get("function") or obj.get("tool")
        args = obj.get("arguments") or obj.get("args") or obj.get("parameters") or {}
        if not name:
            continue
        parsed_calls.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": args if isinstance(args, str) else json.dumps(args, ensure_ascii=False),
                },
            }
        )
        stripped_content = stripped_content.replace(match.group(0), "")

    if parsed_calls:
        response = {**response}
        response["tool_calls"] = existing_calls + parsed_calls
        response["content"] = stripped_content.strip()
        response.setdefault("metadata", {})
        if isinstance(response["metadata"], dict):
            response["metadata"] = {
                **response["metadata"],
                f"{source or 'tagged'}_parsed_calls": len(parsed_calls),
            }
    return response


# --------------------------------------------------------------------------- #
# Registry                                                                     #
# --------------------------------------------------------------------------- #
_REGISTRY = {
    "raw": RawHarness,
    "hermes": HermesHarness,
    "qwen": QwenHarness,
    "pi": PiHarness,
}


def list_harnesses() -> list[str]:
    return sorted(_REGISTRY)


def get_harness(name: str = "raw"):
    try:
        return _REGISTRY[name]()
    except KeyError as exc:
        available = ", ".join(list_harnesses())
        raise ValueError(f"Unsupported harness: {name}. Available: {available}") from exc
