from __future__ import annotations

import pytest

from bench_loop.dashboard.api.routes import chat as chat_routes
from bench_loop.dashboard.api.routes import models
from bench_loop.providers import openai_compat


def test_openai_endpoint_detection_by_port_and_host():
    assert models._is_openai_compat_endpoint("http://127.0.0.1:8088")
    assert models._is_openai_compat_endpoint("http://localhost:11451")
    assert models._is_openai_compat_endpoint("https://openrouter.ai/api")
    assert models._is_openai_compat_endpoint("https://api.openai.com/v1")
    assert not models._is_openai_compat_endpoint("http://localhost:11434")


def test_endpoint_specific_openai_key_takes_precedence(monkeypatch):
    monkeypatch.setenv(
        "BENCHLOOP_OPENAI_KEYS",
        "http://127.0.0.1:8000=sk-local,https://openrouter.ai/api=sk-openrouter",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-global")

    assert openai_compat._api_key_for_endpoint("http://127.0.0.1:8000/") == "sk-local"
    assert openai_compat._api_key_for_endpoint("https://openrouter.ai/api") == "sk-openrouter"
    assert openai_compat._api_key_for_endpoint("http://127.0.0.1:9000") == "sk-global"


@pytest.mark.asyncio
async def test_preflight_uses_openai_chat_completions_for_openai_endpoint(monkeypatch):
    calls: list[dict] = []

    async def fail_ollama_version(endpoint: str):  # pragma: no cover - should not be called
        raise AssertionError("OpenAI-compatible preflight must not check Ollama version")

    class Response:
        status_code = 200
        text = ""

    class Client:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json=None, headers=None):
            calls.append({"url": url, "json": json, "headers": headers})
            return Response()

    monkeypatch.setattr(models, "_fetch_ollama_version", fail_ollama_version)
    monkeypatch.setattr(models.httpx, "AsyncClient", Client)

    result = await models.preflight_model(
        endpoint="http://127.0.0.1:8088",
        model="stepfun-ai/Step-3.7-Flash",
    )

    assert result["ok"] is True
    assert calls[0]["url"] == "http://127.0.0.1:8088/v1/chat/completions"
    assert calls[0]["json"]["max_tokens"] == 16
    assert "options" not in calls[0]["json"]


@pytest.mark.asyncio
async def test_chat_route_forwards_openai_endpoint_auth_headers(monkeypatch):
    monkeypatch.setenv("BENCHLOOP_OPENAI_KEYS", "http://127.0.0.1:8088=sk-local")
    calls: list[dict] = []

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }

    class Client:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json=None, headers=None):
            calls.append({"url": url, "json": json, "headers": headers})
            return Response()

    monkeypatch.setattr(chat_routes.httpx, "AsyncClient", Client)

    result = await chat_routes.chat_generate(
        chat_routes.ChatRequest(
            model="local-model",
            endpoint="http://127.0.0.1:8088",
            provider="openai_compat",
            prompt="Say ok",
        )
    )

    assert result["message"]["content"] == "ok"
    assert calls[0]["url"] == "http://127.0.0.1:8088/v1/chat/completions"
    assert calls[0]["headers"]["Authorization"] == "Bearer sk-local"
