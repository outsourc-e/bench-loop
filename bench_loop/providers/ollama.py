"""Ollama provider implementation."""
from __future__ import annotations

from time import perf_counter
from typing import Any

import httpx


# Long generations (27B+ at high context) routinely take > 60s on consumer GPUs.
# Bumping to 600s + adding granular read timeout via httpx.Timeout so concurrent
# benchmark requests do not get killed mid-response by httpx's default.
TIMEOUT_SECONDS = 600.0
_HTTP_TIMEOUT = httpx.Timeout(connect=15.0, read=600.0, write=60.0, pool=60.0)


async def list_models(endpoint: str) -> list[str]:
    base_url = endpoint.rstrip("/")
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        response = await client.get(f"{base_url}/api/tags")
        response.raise_for_status()
    payload = response.json()
    return [model.get("name", "") for model in payload.get("models", []) if model.get("name")]


async def get_system_info(endpoint: str) -> dict[str, Any]:
    """Try to get hardware info from Ollama.

    Ollama does not currently expose a stable direct hardware API here, so for now
    we return the endpoint for remote identification.
    """
    return {"endpoint": endpoint}


async def chat(endpoint: str, model: str, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
    base_url = endpoint.rstrip("/")
    request_messages = [dict(message) for message in messages]

    payload: dict[str, Any] = {
        "model": model,
        "messages": request_messages,
        "stream": False,
    }
    if kwargs:
        options = {
            key: value
            for key, value in kwargs.items()
            if key
            in {
                "temperature",
                "top_p",
                "top_k",
                "repeat_penalty",
                "seed",
                "num_predict",
                "stop",
            }
            and value is not None
        }
        if "max_tokens" in kwargs and kwargs["max_tokens"] is not None:
            options["num_predict"] = kwargs["max_tokens"]
        if options:
            payload["options"] = options
        for passthrough_key in ("format", "keep_alive", "think"):
            if passthrough_key in kwargs and kwargs[passthrough_key] is not None:
                payload[passthrough_key] = kwargs[passthrough_key]
        if "tools" in kwargs and kwargs["tools"]:
            payload["tools"] = kwargs["tools"]

    if model.startswith("qwen3") and "think" not in payload:
        payload["think"] = False

    started = perf_counter()
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        response = await client.post(f"{base_url}/api/chat", json=payload)
        response.raise_for_status()
    elapsed_ms = (perf_counter() - started) * 1000.0

    body = response.json()
    message = body.get("message") or {}
    content = message.get("content") or message.get("thinking") or ""
    tool_calls = message.get("tool_calls") or []

    eval_count = int(body.get("eval_count") or 0)
    eval_duration = int(body.get("eval_duration") or 0)
    prompt_eval_count = int(body.get("prompt_eval_count") or 0)
    prompt_eval_duration = int(body.get("prompt_eval_duration") or 0)
    load_duration = int(body.get("load_duration") or 0)
    total_duration = int(body.get("total_duration") or 0)

    generation_tok_per_sec = (
        eval_count / (eval_duration / 1_000_000_000) if eval_count > 0 and eval_duration > 0 else 0.0
    )
    prompt_tok_per_sec = (
        prompt_eval_count / (prompt_eval_duration / 1_000_000_000)
        if prompt_eval_count > 0 and prompt_eval_duration > 0
        else 0.0
    )

    total_ms = total_duration / 1_000_000 if total_duration > 0 else elapsed_ms
    ttft_ms = (load_duration + prompt_eval_duration) / 1_000_000 if (load_duration or prompt_eval_duration) else 0.0

    return {
        "content": content,
        "tool_calls": tool_calls,
        "raw_response": body,
        "tokens_prompt": prompt_eval_count,
        "tokens_generated": eval_count,
        "ttft_ms": ttft_ms,
        "total_ms": total_ms,
        "model": body.get("model") or model,
        "eval_count": eval_count,
        "eval_duration": eval_duration,
        "prompt_eval_count": prompt_eval_count,
        "prompt_eval_duration": prompt_eval_duration,
        "load_duration": load_duration,
        "generation_tok_per_sec": generation_tok_per_sec,
        "prompt_eval_tok_per_sec": prompt_tok_per_sec,
        "done_reason": body.get("done_reason", ""),
    }
