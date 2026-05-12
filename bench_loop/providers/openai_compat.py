"""OpenAI-compatible provider implementation."""
from __future__ import annotations

import os
from time import perf_counter
from typing import Any

import httpx


# Long generations (27B+ at high context) routinely take > 60s.
TIMEOUT_SECONDS = 600.0
_HTTP_TIMEOUT = httpx.Timeout(connect=15.0, read=600.0, write=60.0, pool=60.0)


def _auth_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


async def list_models(endpoint: str) -> list[str]:
    base_url = endpoint.rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            response = await client.get(f"{base_url}/v1/models", headers=_auth_headers())
            response.raise_for_status()
    except Exception:
        return []
    payload = response.json()
    return [item.get("id", "") for item in payload.get("data", []) if item.get("id")]


async def get_system_info(endpoint: str) -> dict[str, Any]:
    return {"endpoint": endpoint}


async def chat(endpoint: str, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
    base_url = endpoint.rstrip("/")
    payload: dict[str, Any] = {
        "model": model,
        "messages": [dict(message) for message in messages],
        "max_tokens": int(kwargs.get("max_tokens") or 2048),
        "temperature": float(kwargs.get("temperature") or 0.0),
    }
    # MiniMax / Qwen3 always-reasoning models need temp>0 to avoid loops.
    # Most reasoning servers also recommend top_p/top_k. Forward if provided.
    for k in ("top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty", "frequency_penalty", "stop"):
        if kwargs.get(k) is not None:
            payload[k] = kwargs[k]
    if kwargs.get("tools"):
        payload["tools"] = kwargs["tools"]
    if kwargs.get("response_format"):
        payload["response_format"] = kwargs["response_format"]

    started = perf_counter()
    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            response = await client.post(
                f"{base_url}/v1/chat/completions",
                json=payload,
                headers=_auth_headers(),
            )
            response.raise_for_status()
    except Exception as exc:
        return {
            "content": "",
            "tool_calls": [],
            "raw_response": {},
            "tokens_prompt": 0,
            "tokens_generated": 0,
            "ttft_ms": 0.0,
            "total_ms": (perf_counter() - started) * 1000.0,
            "model": model,
            "prompt_eval_tok_per_sec": 0.0,
            "generation_tok_per_sec": 0.0,
            "error": str(exc),
        }

    elapsed_ms = (perf_counter() - started) * 1000.0
    body = response.json()
    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    usage = body.get("usage") or {}
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    tool_calls = message.get("tool_calls") or []

    # If the model is always-reasoning and emitted *only* a <think> block
    # before hitting max_tokens, content will be empty but reasoning will
    # hold the actual signal. Surface reasoning as content so quality tasks
    # can score against it instead of seeing "". Keeps original content if
    # both fields are populated.
    if not content and reasoning:
        content = reasoning

    completion_tokens = int(usage.get("completion_tokens") or 0)
    prompt_tokens = int(usage.get("prompt_tokens") or 0)

    # Derive an approximate tok/s from wall time when the server doesn't
    # populate explicit speed fields (vmlx/JANGTQ does not). Use total time
    # rather than decode-only since we don't have TTFT here.
    gen_tok_per_sec = 0.0
    if completion_tokens > 0 and elapsed_ms > 0:
        gen_tok_per_sec = completion_tokens / (elapsed_ms / 1000.0)

    return {
        "content": content,
        "tool_calls": tool_calls,
        "raw_response": body,
        "tokens_prompt": prompt_tokens,
        "tokens_generated": completion_tokens,
        "ttft_ms": 0.0,
        "total_ms": elapsed_ms,
        "model": body.get("model") or model,
        "prompt_eval_tok_per_sec": 0.0,
        "generation_tok_per_sec": gen_tok_per_sec,
        "error": "",
        "reasoning_content": reasoning,
    }
