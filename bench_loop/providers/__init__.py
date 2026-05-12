"""Provider registry."""
from __future__ import annotations

from bench_loop.providers import ollama, openai_compat


PROVIDER_REGISTRY = {
    "ollama": ollama,
    "openai_compat": openai_compat,
}


def get_provider(name: str):
    try:
        return PROVIDER_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(PROVIDER_REGISTRY))
        raise ValueError(f"Unsupported provider: {name}. Available: {available}") from exc


__all__ = ["PROVIDER_REGISTRY", "get_provider", "ollama", "openai_compat"]
