"""Model listing — auto-detects local providers (Ollama, LM Studio, oMLX, etc.) and supports remote endpoints."""
from __future__ import annotations

import asyncio
import contextlib
import json
import time

from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx

try:
    from sse_starlette.sse import EventSourceResponse
except Exception:  # pragma: no cover
    EventSourceResponse = None

router = APIRouter()

# Active pull tracking
_active_pulls: dict[str, dict] = {}
_pull_subscribers: dict[str, list[asyncio.Queue]] = {}
_PULL_RETENTION_SECONDS = 300


class PullRequest(BaseModel):
    model: str
    endpoint: str = "http://localhost:11434"


# Common local provider endpoints to probe
KNOWN_PROVIDERS = [
    {"name": "ollama", "url": "http://localhost:11434", "type": "ollama", "label": "Ollama (local)"},
    # Common SSH-tunnel ports for remote Ollama hosts (PC1 / PC2 / lab boxes).
    # We probe them as ollama because that's what people tunnel.
    {"name": "ollama-tunnel-11435", "url": "http://localhost:11435", "type": "ollama", "label": "Ollama @ :11435 (tunnel)"},
    {"name": "ollama-tunnel-11436", "url": "http://localhost:11436", "type": "openai", "label": "vmlx/MLX @ :11436 (tunnel)"},
    {"name": "ollama-tunnel-11437", "url": "http://localhost:11437", "type": "ollama", "label": "Ollama @ :11437 (tunnel)"},
    {"name": "lm-studio", "url": "http://localhost:1234", "type": "openai", "label": "LM Studio (local)"},
    {"name": "omlx", "url": "http://localhost:8000", "type": "openai", "label": "oMLX / Osaurus (local)"},
    {"name": "jan", "url": "http://localhost:1337", "type": "openai", "label": "Jan / Atomic Chat (local)"},
    {"name": "vllm", "url": "http://localhost:8080", "type": "openai", "label": "vLLM (local)"},
    # Any extra endpoints configured by the user via BENCHLOOP_EXTRA_ENDPOINTS
    # (comma-separated url|type|label triples). Appended at probe time below.
]


def _user_extra_providers() -> list[dict]:
    """Read $BENCHLOOP_EXTRA_ENDPOINTS for additional probe targets.

    Format: `url|type|label,url|type|label,...`
    type is `ollama` or `openai`. label is free text.
    """
    import os
    raw = os.environ.get("BENCHLOOP_EXTRA_ENDPOINTS", "").strip()
    if not raw:
        return []
    out: list[dict] = []
    for chunk in raw.split(","):
        parts = [p.strip() for p in chunk.split("|")]
        if len(parts) < 2 or not parts[0]:
            continue
        url = parts[0]
        kind = parts[1] if len(parts) > 1 else "ollama"
        label = parts[2] if len(parts) > 2 else url
        out.append({
            "name": f"extra-{len(out)}",
            "url": url,
            "type": kind,
            "label": label,
        })
    return out


# Minimum Ollama versions known to support specific architectures / quants.
# When the connected Ollama is older, BenchLoop surfaces a preflight warning
# instead of letting the user hit a confusing 500 at run time.
OLLAMA_MIN_VERSIONS = {
    # Qwen 3.5 / 3.6 family
    "qwen35": "0.12.0",
    "qwen36": "0.12.0",
    # TQ (ternary) quantizations
    "tq": "0.12.0",
    # MXFP4
    "mxfp4": "0.11.0",
}


def _version_tuple(v: str) -> tuple[int, ...]:
    cleaned = "".join(c if (c.isdigit() or c == ".") else " " for c in (v or ""))
    parts = [p for p in cleaned.split() if p]
    if not parts:
        return (0,)
    first = parts[0]
    try:
        return tuple(int(x) for x in first.split("."))
    except ValueError:
        return (0,)


def _classify_model_support(name: str, details: dict) -> dict:
    """Return a support flag describing which min-version bucket a model needs."""
    family = (details.get("family") or "").lower()
    quant = (details.get("quantization_level") or "").lower()
    fmt = (details.get("format") or "").lower()
    lname = name.lower()

    required_key = None
    reason = None

    if "qwen35" in family or "qwen3.5" in lname:
        required_key = "qwen35"
        reason = "Qwen 3.5 architecture"
    elif "qwen36" in family or "qwen3.6" in lname:
        required_key = "qwen36"
        reason = "Qwen 3.6 architecture"
    elif "tq" in quant or "tq" in lname or "-tq" in lname:
        required_key = "tq"
        reason = "TQ (ternary) quantization"
    elif "mxfp4" in quant or "mxfp4" in lname:
        required_key = "mxfp4"
        reason = "MXFP4 quantization"

    if not required_key:
        return {"required_version": None, "reason": None}

    return {
        "required_version": OLLAMA_MIN_VERSIONS[required_key],
        "reason": reason,
    }


async def _fetch_ollama_version(endpoint: str) -> str | None:
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            resp = await client.get(f"{endpoint.rstrip('/')}/api/version")
            if resp.status_code == 200:
                return resp.json().get("version")
    except Exception:
        return None
    return None


async def _fetch_ollama_models(endpoint: str) -> list[dict]:
    """Fetch models from an Ollama-style endpoint.

    LM Studio quirk: when pointed at LM Studio's OpenAI-compatible server, the
    request to `/api/tags` returns HTTP 200 with a non-Ollama body and logs
    'Unexpected endpoint or method' on its side. So we must validate the JSON
    shape, not just the status code, to avoid silently treating it as 'ollama
    with zero models'.
    """
    async with httpx.AsyncClient(timeout=5) as client:
        resp = await client.get(f"{endpoint.rstrip('/')}/api/tags")
        resp.raise_for_status()
    body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
    if not isinstance(body, dict) or "models" not in body:
        raise ValueError(f"{endpoint} did not respond like Ollama (no `models` key in /api/tags response)")
    raw = body.get("models", [])
    version = await _fetch_ollama_version(endpoint)
    installed_tuple = _version_tuple(version) if version else (0,)
    models = []
    for m in raw:
        name = m.get("name", "")
        details = m.get("details", {})
        size_bytes = m.get("size", 0)
        size_gb = round(size_bytes / (1024**3), 1) if size_bytes else None
        support = _classify_model_support(name, details)
        required = support["required_version"]
        supported = True
        warning = None
        if required:
            if installed_tuple < _version_tuple(required):
                supported = False
                warning = (
                    f"Requires Ollama {required}+ ({support['reason']}); "
                    f"installed is {version or 'unknown'}. Upgrade with "
                    f"`brew upgrade ollama` then restart `ollama serve`."
                )
        models.append({
            "name": name,
            "size_gb": size_gb,
            "quantization": details.get("quantization_level", ""),
            "family": details.get("family", ""),
            "parameter_size": details.get("parameter_size", ""),
            "format": details.get("format", ""),
            "provider_version": version,
            "supported": supported,
            "warning": warning,
        })
    return models


async def _fetch_openai_models(endpoint: str) -> list[dict]:
    """Fetch models from an OpenAI-compatible endpoint."""
    async with httpx.AsyncClient(timeout=5) as client:
        resp = await client.get(f"{endpoint.rstrip('/')}/v1/models")
        resp.raise_for_status()
    raw = resp.json().get("data", [])
    return [
        {
            "name": m.get("id", ""),
            "size_gb": None,
            "quantization": "",
            "family": "",
            "parameter_size": "",
            "format": "",
        }
        for m in raw
        if m.get("id")
    ]


async def _probe_provider(provider: dict) -> dict | None:
    """Check if a provider is reachable and return its info + models."""
    try:
        if provider["type"] == "ollama":
            models = await _fetch_ollama_models(provider["url"])
        else:
            models = await _fetch_openai_models(provider["url"])
        return {
            "name": provider["name"],
            "label": provider["label"],
            "url": provider["url"],
            "type": provider["type"],
            "available": True,
            "model_count": len(models),
            "models": models,
        }
    except Exception:
        return None


_OPENAI_HINT_PORTS = {1234, 1337, 5001, 8000, 8080, 8081}


@router.get("/models")
async def list_models(endpoint: str = Query(default="")):
    """List models. If endpoint specified, query that. Otherwise auto-detect local providers."""
    if endpoint:
        # If the port is well-known for OpenAI-compatible servers, try that first.
        from urllib.parse import urlparse
        try:
            port = urlparse(endpoint).port
        except Exception:
            port = None
        order: list[str] = ["openai", "ollama"] if port in _OPENAI_HINT_PORTS else ["ollama", "openai"]
        last_error: str | None = None
        for kind in order:
            try:
                if kind == "ollama":
                    models = await _fetch_ollama_models(endpoint)
                else:
                    models = await _fetch_openai_models(endpoint)
                return {
                    "providers": [{
                        "name": "custom",
                        "label": f"Custom ({endpoint})",
                        "url": endpoint,
                        "type": kind,
                        "available": True,
                        "model_count": len(models),
                        "models": models,
                    }],
                    "total_models": len(models),
                }
            except Exception as exc:
                last_error = f"{kind}: {exc}"
        return {
            "providers": [],
            "total_models": 0,
            "error": f"Could not connect to {endpoint} ({last_error})",
        }

    detected = []
    for provider in KNOWN_PROVIDERS + _user_extra_providers():
        result = await _probe_provider(provider)
        if result:
            detected.append(result)

    total = sum(p["model_count"] for p in detected)
    return {
        "providers": detected,
        "total_models": total,
    }


@router.get("/models/preflight")
async def preflight_model(
    endpoint: str = Query(...),
    model: str = Query(...),
):
    """Pre-benchmark diagnostic: can this model actually load on this endpoint?

    Returns structured JSON with:
      - ok: True when the model loads and responds
      - reason: machine-readable reason code ('version_mismatch',
        'missing_blob', 'oom', 'not_found', 'endpoint_unreachable',
        'unknown_failure')
      - message: human-readable explanation with a fix
      - required_version / provider_version: for version_mismatch
      - raw: raw error text truncated for debugging

    Called by the UI before starting a run so users don't burn a benchmark
    slot on a model the server can't actually load.
    """
    ep = endpoint.rstrip("/")

    # Version check against our support table (fast path, no model load needed)
    version = await _fetch_ollama_version(ep)
    installed_tuple = _version_tuple(version) if version else (0,)
    # We don't have the details dict for an arbitrary model name — do a best
    # effort classification from the name alone.
    support = _classify_model_support(model, {"family": "", "quantization_level": ""})
    required = support["required_version"]
    if required and installed_tuple < _version_tuple(required):
        return {
            "ok": False,
            "reason": "version_mismatch",
            "message": (
                f"Model `{model}` needs Ollama {required}+ ({support['reason']}). "
                f"Installed: {version or 'unknown'}. Upgrade with "
                f"`brew upgrade ollama` then restart `ollama serve`."
            ),
            "required_version": required,
            "provider_version": version,
            "raw": None,
        }

    # Actual load test: minimum-cost chat round-trip
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"{ep}/api/chat",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "ok"}],
                    "stream": False,
                    "options": {"num_predict": 1},
                },
            )
    except httpx.HTTPError as exc:
        return {
            "ok": False,
            "reason": "endpoint_unreachable",
            "message": f"Cannot reach {ep}: {exc}",
            "raw": str(exc),
            "provider_version": version,
        }

    if resp.status_code == 200:
        return {
            "ok": True,
            "reason": None,
            "message": "Model loaded and responded.",
            "provider_version": version,
            "raw": None,
        }

    raw = resp.text[:600]
    lower = raw.lower()

    if "no such file or directory" in lower:
        return {
            "ok": False,
            "reason": "missing_blob",
            "message": (
                f"Model file missing on disk for `{model}`. The registry references a "
                f"blob that is not present. Fix: `ollama rm {model} && ollama pull {model}`."
            ),
            "raw": raw,
            "provider_version": version,
        }

    if "out of memory" in lower or "cudamalloc" in lower:
        return {
            "ok": False,
            "reason": "oom",
            "message": (
                f"`{model}` failed to load due to VRAM pressure. Free up memory "
                f"(`ollama stop --all` on other hosts), lower context, or use a smaller quant."
            ),
            "raw": raw,
            "provider_version": version,
        }

    if "model" in lower and "not found" in lower:
        return {
            "ok": False,
            "reason": "not_found",
            "message": f"`{model}` is not installed at {ep}. Fix: `ollama pull {model}`.",
            "raw": raw,
            "provider_version": version,
        }

    if "unable to load model" in lower or "failed to load model" in lower:
        return {
            "ok": False,
            "reason": "version_mismatch",
            "message": (
                f"Ollama could not load `{model}`. This usually means the installed Ollama "
                f"version is older than the model's architecture or quantization format "
                f"(e.g. Qwen3.5/3.6, TQ1_0/TQ2_0, MXFP4). Upgrade with "
                f"`brew upgrade ollama` (target 0.12+), then restart `ollama serve`."
            ),
            "raw": raw,
            "provider_version": version,
        }

    return {
        "ok": False,
        "reason": "unknown_failure",
        "message": f"Health check failed ({resp.status_code}): {raw[:300]}",
        "raw": raw,
        "provider_version": version,
    }


@router.get("/models/search-hf")
async def search_huggingface(
    query: str = Query(default=""),
    limit: int = Query(default=20),
    page: int = Query(default=1),
    format_filter: str = Query(default="", alias="format"),
):
    """Search HuggingFace for trending/popular models with filtering and pagination."""
    try:
        fetch_limit = limit * 3 if format_filter else limit
        offset = (page - 1) * limit

        async with httpx.AsyncClient(timeout=10) as client:
            params = {
                "sort": "trendingScore" if not query else "downloads",
                "direction": "-1",
                "limit": fetch_limit + offset,
                "pipeline_tag": "text-generation",
            }
            if query:
                params["search"] = query
            resp = await client.get("https://huggingface.co/api/models", params=params)
            resp.raise_for_status()

        format_keywords = {
            "gguf": ["gguf"],
            "mlx": ["mlx"],
            "gptq": ["gptq"],
            "awq": ["awq"],
            "fp16": ["fp16", "float16"],
            "fp8": ["fp8"],
            "bf16": ["bf16", "bfloat16"],
        }

        models = []
        for m in resp.json():
            model_id = m.get("id", "")
            if not model_id or "/" not in model_id:
                continue
            downloads = m.get("downloads", 0)
            likes = m.get("likes", 0)
            tags = m.get("tags", [])
            pipeline_tag = m.get("pipeline_tag", "")
            author = model_id.split("/")[0]
            model_name = model_id.split("/")[-1]

            detected_formats = []
            searchable = " ".join(tags + [model_name]).lower()
            for fmt, keywords in format_keywords.items():
                if any(kw in searchable for kw in keywords):
                    detected_formats.append(fmt)
            if not detected_formats:
                detected_formats.append("safetensors")

            if format_filter and format_filter.lower() not in [f.lower() for f in detected_formats]:
                continue

            models.append({
                "id": model_id,
                "author": author,
                "name": model_name,
                "avatar_url": f"https://huggingface.co/api/organizations/{author}/avatar",
                "downloads": downloads,
                "likes": likes,
                "pipeline_tag": pipeline_tag,
                "tags": tags[:15],
                "formats": detected_formats,
                "url": f"https://huggingface.co/{model_id}",
                "created_at": m.get("createdAt", ""),
            })

        total = len(models)
        page_models = models[offset:offset + limit]

        return {
            "query": query,
            "format_filter": format_filter,
            "models": page_models,
            "count": len(page_models),
            "total": total,
            "page": page,
            "pages": (total + limit - 1) // limit if total > 0 else 1,
        }
    except Exception as exc:
        return {
            "query": query,
            "models": [],
            "count": 0,
            "total": 0,
            "page": 1,
            "pages": 1,
            "error": str(exc),
        }


def _is_pull_done(info: dict | None) -> bool:
    return bool(info and info.get("done"))


async def _broadcast_pull_update(pull_id: str) -> None:
    info = _active_pulls.get(pull_id)
    if not info:
        return
    for queue in list(_pull_subscribers.get(pull_id, [])):
        with contextlib.suppress(asyncio.QueueFull):
            queue.put_nowait(dict(info))


async def _cleanup_pull_after_delay(pull_id: str, delay: int = _PULL_RETENTION_SECONDS) -> None:
    await asyncio.sleep(delay)
    _active_pulls.pop(pull_id, None)
    _pull_subscribers.pop(pull_id, None)


async def _do_pull(pull_id: str, model_name: str, endpoint: str):
    """Background task: pull model from Ollama and update _active_pulls."""
    info = _active_pulls[pull_id]
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10, read=None, write=30, pool=30)) as client:
            async with client.stream(
                "POST",
                f"{endpoint.rstrip('/')}/api/pull",
                json={"name": model_name, "stream": True},
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    status = chunk.get("status") or info.get("status") or "pulling"
                    completed = chunk.get("completed")
                    total = chunk.get("total")
                    progress = info.get("progress", 0)

                    if isinstance(completed, (int, float)) and isinstance(total, (int, float)) and total > 0:
                        progress = round((completed / total) * 100, 1)
                    elif status == "success":
                        progress = 100

                    info.update({
                        "status": status,
                        "progress": max(0, min(progress, 100)),
                        "completed": completed,
                        "total": total,
                        "digest": chunk.get("digest"),
                        "error": chunk.get("error"),
                        "done": status == "success" or bool(chunk.get("error")),
                        "updated_at": time.time(),
                    })

                    if info["done"]:
                        info["finished_at"] = time.time()
                        if chunk.get("error"):
                            info["status"] = "error"

                    await _broadcast_pull_update(pull_id)

        if not info.get("done"):
            info.update({
                "status": "success",
                "progress": 100,
                "done": True,
                "updated_at": time.time(),
                "finished_at": time.time(),
            })
            await _broadcast_pull_update(pull_id)
    except Exception as exc:
        info.update({
            "status": "error",
            "error": str(exc),
            "done": True,
            "updated_at": time.time(),
            "finished_at": time.time(),
        })
        await _broadcast_pull_update(pull_id)
    finally:
        asyncio.create_task(_cleanup_pull_after_delay(pull_id))


@router.post("/models/pull")
async def pull_model(body: PullRequest):
    """Start pulling a model via Ollama. Returns pull_id for tracking."""
    model_name = body.model.strip()
    endpoint = body.endpoint.strip().rstrip("/")
    if not model_name:
        return {"error": "model name required"}

    pull_id = f"pull-{int(time.time())}-{model_name.replace('/', '-').replace(':', '-')}"
    _active_pulls[pull_id] = {
        "pull_id": pull_id,
        "model": model_name,
        "endpoint": endpoint,
        "status": "starting",
        "progress": 0,
        "completed": 0,
        "total": 0,
        "digest": None,
        "error": None,
        "done": False,
        "started_at": time.time(),
        "updated_at": time.time(),
    }
    asyncio.create_task(_do_pull(pull_id, model_name, endpoint))
    return {"pull_id": pull_id, "model": model_name}


@router.get("/models/pull/active")
async def active_pulls():
    """List all active and recent pulls."""
    now = time.time()
    stale = [
        key for key, value in _active_pulls.items()
        if value.get("done") and value.get("finished_at") and now - value["finished_at"] > _PULL_RETENTION_SECONDS
    ]
    for key in stale:
        _active_pulls.pop(key, None)
        _pull_subscribers.pop(key, None)
    pulls = sorted(_active_pulls.values(), key=lambda item: item.get("started_at", 0), reverse=True)
    return {"pulls": pulls}


@router.get("/models/pull/{pull_id}")
async def pull_status(pull_id: str):
    """Get status of a specific pull."""
    info = _active_pulls.get(pull_id)
    if not info:
        return {"error": "unknown pull_id"}
    return info


async def _sse_event_generator(pull_id: str, request: Request):
    info = _active_pulls.get(pull_id)
    if not info:
        yield {"event": "error", "data": json.dumps({"error": "unknown pull_id"})}
        return

    queue: asyncio.Queue = asyncio.Queue(maxsize=20)
    _pull_subscribers.setdefault(pull_id, []).append(queue)
    try:
        yield {"event": "pull", "data": json.dumps(dict(info))}
        while True:
            if await request.is_disconnected():
                break
            if _is_pull_done(_active_pulls.get(pull_id)) and queue.empty():
                final_info = _active_pulls.get(pull_id)
                if final_info:
                    yield {"event": "pull", "data": json.dumps(dict(final_info))}
                break
            try:
                update = await asyncio.wait_for(queue.get(), timeout=15)
                yield {"event": "pull", "data": json.dumps(update)}
            except asyncio.TimeoutError:
                heartbeat = _active_pulls.get(pull_id)
                if heartbeat:
                    yield {"event": "ping", "data": json.dumps({"pull_id": pull_id, "status": heartbeat.get("status")})}
                else:
                    break
    finally:
        subscribers = _pull_subscribers.get(pull_id, [])
        with contextlib.suppress(ValueError):
            subscribers.remove(queue)
        if not subscribers:
            _pull_subscribers.pop(pull_id, None)


@router.get("/models/pull/{pull_id}/stream")
async def pull_stream(pull_id: str, request: Request):
    """SSE stream of pull progress."""
    if pull_id not in _active_pulls:
        return {"error": "unknown pull_id"}

    if EventSourceResponse:
        return EventSourceResponse(_sse_event_generator(pull_id, request))

    async def plain_stream():
        async for event in _sse_event_generator(pull_id, request):
            yield f"event: {event['event']}\ndata: {event['data']}\n\n"

    return StreamingResponse(plain_stream(), media_type="text/event-stream")


@router.get('/models/hf-details')
async def hf_model_details(repo: str = Query(...)):
    """Fetch detailed HF repo info including GGUF file sizes for fit checks."""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            detail_resp = await client.get(f'https://huggingface.co/api/models/{repo}')
            detail_resp.raise_for_status()
            tree_resp = await client.get(f'https://huggingface.co/api/models/{repo}/tree/main')
            tree_resp.raise_for_status()

        detail = detail_resp.json()
        files = tree_resp.json()

        gguf_files = []
        total_size = 0
        for f in files:
            path = f.get('path', '')
            size = f.get('size') or 0
            if path.lower().endswith('.gguf'):
                gguf_files.append({
                    'path': path,
                    'size_bytes': size,
                    'size_gb': round(size / (1024**3), 2) if size else None,
                })
                total_size += size

        largest = max(gguf_files, key=lambda x: x.get('size_bytes') or 0) if gguf_files else None

        return {
            'repo': repo,
            'id': detail.get('id', repo),
            'downloads': detail.get('downloads', 0),
            'likes': detail.get('likes', 0),
            'tags': detail.get('tags', []),
            'cardData': detail.get('cardData', {}),
            'gguf_files': gguf_files,
            'largest_gguf': largest,
            'total_gguf_size_gb': round(total_size / (1024**3), 2) if total_size else None,
        }
    except Exception as exc:
        return {
            'repo': repo,
            'error': str(exc),
            'gguf_files': [],
            'largest_gguf': None,
            'total_gguf_size_gb': None,
        }
