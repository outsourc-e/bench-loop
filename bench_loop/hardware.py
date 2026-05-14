"""Hardware detection helpers."""
from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from urllib.parse import urlparse
from uuid import getnode

import psutil


def _run_command(command: list[str]) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
    except Exception:
        return ""
    return (result.stdout or "").strip()


def _detect_cpu_model() -> str:
    system = platform.system()
    if system == "Darwin":
        brand = _run_command(["sysctl", "-n", "machdep.cpu.brand_string"])
        if brand:
            return brand
        hardware = _run_command(["system_profiler", "SPHardwareDataType"])
        match = re.search(r"Chip:\s*(.+)", hardware)
        if match:
            return match.group(1).strip()
    if system == "Linux":
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as handle:
                content = handle.read()
            for pattern in (r"model name\s*:\s*(.+)", r"Hardware\s*:\s*(.+)"):
                match = re.search(pattern, content)
                if match:
                    return match.group(1).strip()
        except Exception:
            pass
    processor = platform.processor() or platform.uname().processor
    machine = platform.machine()
    return processor or machine or "unknown"


def _detect_apple_gpu() -> dict[str, object]:
    """Apple Silicon GPU is part of the SoC; report the chip as GPU-ish hardware."""
    if platform.system() != "Darwin":
        return {}
    hardware = _run_command(["system_profiler", "SPHardwareDataType"])
    chip = ""
    match = re.search(r"Chip:\s*(.+)", hardware)
    if match:
        chip = match.group(1).strip()
    memory_gb = round(psutil.virtual_memory().total / (1024**3), 2)
    if chip:
        return {
            "gpu": f"{chip} integrated GPU",
            "gpu_memory_gb": memory_gb,
            "gpu_temperature_c": None,
            "gpu_details": [{"name": f"{chip} integrated GPU", "memory_gb": memory_gb, "memory_type": "unified"}],
        }
    return {}


def _env_float(name: str, default: float = 0.0) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _env_hardware_overrides() -> dict[str, object]:
    """Explicit hardware attribution for tunnels/remote endpoints.

    Useful when benchmarking a server through localhost:PORT where the CLI
    process is local but inference runs elsewhere.
    """
    overrides: dict[str, object] = {}
    if os.environ.get("BENCHLOOP_HARDWARE_LABEL"):
        overrides["hardware_label"] = os.environ["BENCHLOOP_HARDWARE_LABEL"]
    if os.environ.get("BENCHLOOP_GPU"):
        overrides["gpu"] = os.environ["BENCHLOOP_GPU"]
    if os.environ.get("BENCHLOOP_CPU"):
        overrides["cpu"] = os.environ["BENCHLOOP_CPU"]
    if os.environ.get("BENCHLOOP_GPU_MEMORY_GB"):
        overrides["gpu_memory_gb"] = _env_float("BENCHLOOP_GPU_MEMORY_GB")
    if os.environ.get("BENCHLOOP_SYSTEM_MEMORY_GB"):
        overrides["system_memory_gb"] = _env_float("BENCHLOOP_SYSTEM_MEMORY_GB")
    return overrides


def _detect_gpu() -> dict[str, object]:
    overrides = _env_hardware_overrides()
    if overrides.get("gpu") or overrides.get("hardware_label"):
        return {
            "gpu": str(overrides.get("gpu", "")),
            "gpu_memory_gb": float(overrides.get("gpu_memory_gb", 0.0) or 0.0),
            "gpu_temperature_c": None,
            "gpu_details": [],
            "hardware_label": str(overrides.get("hardware_label", "")),
        }

    if not shutil.which("nvidia-smi"):
        apple = _detect_apple_gpu()
        if apple:
            return apple
        return {
            "gpu": "",
            "gpu_memory_gb": 0.0,
            "gpu_temperature_c": None,
            "gpu_details": [],
        }

    query = "name,memory.total,temperature.gpu"
    output = _run_command(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"])
    if not output:
        return {
            "gpu": "",
            "gpu_memory_gb": 0.0,
            "gpu_temperature_c": None,
            "gpu_details": [],
        }

    details: list[dict[str, object]] = []
    total_memory_gb = 0.0
    temperatures: list[float] = []
    names: list[str] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        name, memory_mb, temp_c = parts[0], parts[1], parts[2]
        memory_gb = float(memory_mb) / 1024 if memory_mb else 0.0
        temperature = float(temp_c) if temp_c else None
        details.append({"name": name, "memory_gb": memory_gb, "temperature_c": temperature})
        names.append(name)
        total_memory_gb += memory_gb
        if temperature is not None:
            temperatures.append(temperature)

    return {
        "gpu": ", ".join(names),
        "gpu_memory_gb": round(total_memory_gb, 2),
        "gpu_temperature_c": round(max(temperatures), 1) if temperatures else None,
        "gpu_details": details,
    }


def _endpoint_host(endpoint: str | None) -> str:
    if not endpoint:
        return ""
    parsed = urlparse(endpoint if "://" in endpoint else f"http://{endpoint}")
    return (parsed.hostname or "").strip().lower()


def _is_local_endpoint(endpoint: str | None) -> bool:
    """True only if endpoint is the *default* local Ollama port.

    Non-default localhost ports (11435, 11436, 1234, 8000, 8080, etc.) are
    treated as remote because they're commonly used for SSH/tcp tunnels to
    other machines. Better to under-claim local than misreport hardware.
    """
    if not endpoint:
        return True
    parsed = urlparse(endpoint if "://" in endpoint else f"http://{endpoint}")
    host = (parsed.hostname or "").strip().lower()
    port = parsed.port
    if host not in {"", "localhost", "127.0.0.1", "::1"}:
        return False
    # Treat the standard Ollama port (11434), LM Studio (1234), MLX (8000),
    # llama.cpp alternates (8080/8081), Jan (1337), and common OpenAI-compat
    # local ports as truly local. Any other localhost port is still more likely
    # to be a tunnel than the actual model host.
    return port in {None, 11434, 1234, 1337, 5001, 8000, 8080, 8081}


def _probe_remote_ollama_hardware(endpoint: str) -> dict[str, object] | None:
    """Best-effort probe of a remote Ollama host for CPU/GPU/VRAM info.

    Sources:
      - `/api/ps` — running models with `size_vram` (gives us a floor on GPU memory)
      - `/api/version` — ollama version
      - response headers and any other metadata available
    """
    import json
    from urllib.request import Request, urlopen
    from urllib.error import URLError

    base = endpoint.rstrip("/")
    info: dict[str, object] = {}
    try:
        req = Request(f"{base}/api/ps")
        with urlopen(req, timeout=2) as r:
            data = json.loads(r.read().decode("utf-8"))
        max_vram_bytes = 0
        for m in data.get("models", []) or []:
            size_vram = m.get("size_vram") or 0
            if size_vram > max_vram_bytes:
                max_vram_bytes = size_vram
        if max_vram_bytes:
            # Reported VRAM is what the loaded model is consuming — it's a lower
            # bound on total GPU memory, not the full card size. Tag it clearly.
            info["gpu_memory_gb"] = round(max_vram_bytes / (1024 ** 3), 2)
            info["gpu_memory_note"] = "in-use (lower bound)"
    except (URLError, OSError, ValueError, TimeoutError):
        pass
    try:
        req = Request(f"{base}/api/version")
        with urlopen(req, timeout=2) as r:
            data = json.loads(r.read().decode("utf-8"))
        if data.get("version"):
            info["backend_version"] = f"ollama {data['version']}"
    except (URLError, OSError, ValueError, TimeoutError):
        pass
    return info or None


def detect_hardware(endpoint: str | None = None) -> dict[str, object]:
    virtual_memory = psutil.virtual_memory()
    logical_cores = psutil.cpu_count(logical=True) or 0
    physical_cores = psutil.cpu_count(logical=False) or logical_cores
    is_remote = bool(endpoint) and not _is_local_endpoint(endpoint)
    endpoint_host = _endpoint_host(endpoint)

    if is_remote:
        # Don't lie: the local CPU/GPU is NOT what's running the model. For
        # tunnels, prefer explicit BENCHLOOP_* hardware labels. Otherwise use
        # best-effort remote probe and mark unknown fields blank.
        remote = _probe_remote_ollama_hardware(endpoint or "") or {}
        overrides = _env_hardware_overrides()
        gpu_info = {
            "gpu": str(overrides.get("gpu") or remote.get("gpu", "")),
            "gpu_memory_gb": float(overrides.get("gpu_memory_gb") or remote.get("gpu_memory_gb", 0.0) or 0.0),
            "gpu_temperature_c": None,
            "gpu_details": [],
        }
        hardware_label = str(overrides.get("hardware_label") or "")
        cpu = str(overrides.get("cpu") or "")
        system_memory_gb = float(overrides.get("system_memory_gb") or 0.0)
        machine_id = f"remote:{endpoint_host}" if endpoint_host else "remote"
        return {
            "machine_id": machine_id,
            "os": "remote",
            "platform": f"remote@{endpoint_host}",
            "architecture": "",
            "cpu": cpu,
            "cpu_logical_cores": 0,
            "cpu_physical_cores": 0,
            "system_memory_gb": system_memory_gb,
            "backend": "ollama",
            "endpoint": endpoint or "",
            "is_remote": True,
            "remote_host": endpoint_host,
            "hardware_label": hardware_label,
            **gpu_info,
        }

    gpu_info = _detect_gpu()

    # Stable per-machine id derived from MAC address (hex). Kept as machine_id
    # for dedupe, but display layers should show cpu/gpu/ram, not the hex.
    machine_id = hex(getnode())

    return {
        "machine_id": machine_id,
        "os": platform.system(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "cpu": _detect_cpu_model(),
        "cpu_logical_cores": logical_cores,
        "cpu_physical_cores": physical_cores,
        "system_memory_gb": round(virtual_memory.total / (1024**3), 2),
        "backend": "ollama",
        "endpoint": endpoint or "http://localhost:11434",
        "is_remote": False,
        **gpu_info,
    }
