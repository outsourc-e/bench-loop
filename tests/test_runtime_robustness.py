from __future__ import annotations

import pytest

from bench_loop import hardware
from bench_loop.dashboard.api.routes import benchmark


def test_nvidia_smi_na_values_do_not_crash_gpu_detection(monkeypatch):
    monkeypatch.delenv("BENCHLOOP_GPU", raising=False)
    monkeypatch.delenv("BENCHLOOP_HARDWARE_LABEL", raising=False)
    monkeypatch.setattr(hardware.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        hardware,
        "_run_command",
        lambda command: "NVIDIA GB10 Grace Blackwell, [N/A], [N/A]",
    )

    result = hardware._detect_gpu()

    assert result["gpu"] == "NVIDIA GB10 Grace Blackwell"
    assert result["gpu_memory_gb"] == 0.0
    assert result["gpu_temperature_c"] is None
    assert result["gpu_details"] == [
        {"name": "NVIDIA GB10 Grace Blackwell", "memory_gb": 0.0, "temperature_c": None}
    ]


@pytest.mark.asyncio
async def test_active_run_response_omits_internal_task(monkeypatch):
    internal_task = object()
    monkeypatch.setitem(
        benchmark._active_runs,
        "active123",
        {
            "run_id": "active123",
            "status": "running",
            "task": internal_task,
            "events": [],
        },
    )

    result = await benchmark.get_run("active123")

    assert result["status"] == "running"
    assert "task" not in result
    assert benchmark._active_runs["active123"]["task"] is internal_task
