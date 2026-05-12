"""Hardware detection helpers."""
from __future__ import annotations

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


def _detect_gpu() -> dict[str, object]:
    if not shutil.which("nvidia-smi"):
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
    host = _endpoint_host(endpoint)
    return host in {"", "localhost", "127.0.0.1", "::1"}


def _probe_remote_ollama_hardware(endpoint: str) -> dict[str, object] | None:
    """Best-effort probe of a remote Ollama host for CPU/GPU info.

    Ollama exposes `/api/ps` (running models with GPU info) and `/api/show`
    metadata. Neither gives us full CPU info, so this returns whatever we can
    extract; missing fields stay blank rather than lying with the local Mac's CPU.
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
        for m in data.get("models", []) or []:
            details = m.get("details") or {}
            family = details.get("family") or ""
            size_vram = m.get("size_vram") or 0
            if size_vram and not info.get("gpu_memory_gb"):
                info["gpu_memory_gb"] = round(size_vram / (1024 ** 3), 2)
            if family and not info.get("_seen_family"):
                info["_seen_family"] = family
        info.pop("_seen_family", None)
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
        # Don't lie: the local CPU/GPU is NOT what's running the model. Blank
        # out hardware fields and label by host. Try a best-effort remote probe.
        remote = _probe_remote_ollama_hardware(endpoint or "") or {}
        gpu_info = {
            "gpu": str(remote.get("gpu", "")),
            "gpu_memory_gb": float(remote.get("gpu_memory_gb", 0.0) or 0.0),
            "gpu_temperature_c": None,
            "gpu_details": [],
        }
        machine_id = f"remote:{endpoint_host}" if endpoint_host else "remote"
        return {
            "machine_id": machine_id,
            "os": "remote",
            "platform": f"remote@{endpoint_host}",
            "architecture": "",
            "cpu": "",  # honest: we don't know remote CPU
            "cpu_logical_cores": 0,
            "cpu_physical_cores": 0,
            "system_memory_gb": 0.0,
            "backend": "ollama",
            "endpoint": endpoint or "",
            "is_remote": True,
            "remote_host": endpoint_host,
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
