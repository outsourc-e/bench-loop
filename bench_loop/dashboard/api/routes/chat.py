"""Chat — quick smoke-test against a local model with tok/s metrics."""
from __future__ import annotations

import time
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from bench_loop.providers.openai_compat import _auth_headers as _openai_headers

router = APIRouter()


class ChatRequest(BaseModel):
    model: str
    endpoint: str = "http://localhost:11434"
    provider: str = "ollama"
    prompt: str
    system: str | None = None


@router.post("/chat/generate")
async def chat_generate(req: ChatRequest):
    if not req.model:
        raise HTTPException(status_code=400, detail="model is required")
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt is required")

    started = time.perf_counter()

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            if req.provider == "ollama":
                payload: dict[str, Any] = {
                    "model": req.model,
                    "prompt": req.prompt,
                    "stream": False,
                }
                if req.system:
                    payload["system"] = req.system
                resp = await client.post(f"{req.endpoint.rstrip('/')}/api/generate", json=payload)
                resp.raise_for_status()
                data = resp.json()
                content = data.get("response", "")
                prompt_tokens = data.get("prompt_eval_count", 0) or 0
                completion_tokens = data.get("eval_count", 0) or 0
                eval_ns = data.get("eval_duration", 0) or 0
                prompt_ns = data.get("prompt_eval_duration", 0) or 0
                tok_per_s = (completion_tokens / (eval_ns / 1e9)) if eval_ns else 0.0
                ttft_ms = (prompt_ns / 1e6) if prompt_ns else 0.0
            else:
                # openai_compat
                payload = {
                    "model": req.model,
                    "messages": (
                        ([{"role": "system", "content": req.system}] if req.system else [])
                        + [{"role": "user", "content": req.prompt}]
                    ),
                    "stream": False,
                }
                resp = await client.post(
                    f"{req.endpoint.rstrip('/')}/v1/chat/completions",
                    json=payload,
                    headers=_openai_headers(req.endpoint),
                )
                resp.raise_for_status()
                data = resp.json()
                content = (
                    (data.get("choices", [{}])[0].get("message", {}) or {}).get("content", "")
                )
                usage = data.get("usage", {}) or {}
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                elapsed = max(time.perf_counter() - started, 1e-6)
                tok_per_s = (completion_tokens / elapsed) if completion_tokens else 0.0
                ttft_ms = 0.0  # not reliably available without streaming
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"upstream error: {exc}") from exc

    latency_ms = (time.perf_counter() - started) * 1000.0

    return {
        "message": {"role": "assistant", "content": content},
        "metrics": {
            "latencyMs": latency_ms,
            "ttftMs": ttft_ms,
            "promptTokens": prompt_tokens,
            "completionTokens": completion_tokens,
            "tokensPerSecond": tok_per_s,
            "model": req.model,
            "provider": req.provider,
            "endpoint": req.endpoint,
            "harness": "raw",
        },
    }
