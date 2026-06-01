import asyncio

import pytest
from fastapi import HTTPException

from bench_loop.dashboard.api.routes import benchmark


def run(coro):
    return asyncio.run(coro)


def test_delete_run_removes_persisted_run_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "RUNS_DIR", tmp_path)
    run_dir = tmp_path / "20260601-120000-model-local-ollama"
    run_dir.mkdir()
    (run_dir / "run.json").write_text("{}", encoding="utf-8")

    result = run(benchmark.delete_run(run_dir.name))

    assert result == {"ok": True, "run_id": run_dir.name}
    assert not run_dir.exists()


def test_delete_run_rejects_path_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "RUNS_DIR", tmp_path)

    with pytest.raises(HTTPException) as exc:
        run(benchmark.delete_run("../outside"))

    assert exc.value.status_code == 400


def test_delete_run_rejects_active_run(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "RUNS_DIR", tmp_path)
    benchmark._active_runs["active123"] = {"status": "running"}

    try:
        with pytest.raises(HTTPException) as exc:
            run(benchmark.delete_run("active123"))
        assert exc.value.status_code == 409
    finally:
        benchmark._active_runs.pop("active123", None)


def test_delete_run_returns_404_for_missing_run(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "RUNS_DIR", tmp_path)

    with pytest.raises(HTTPException) as exc:
        run(benchmark.delete_run("missing"))

    assert exc.value.status_code == 404
