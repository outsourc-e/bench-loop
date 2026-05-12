"""Benchmark suite registry."""
from __future__ import annotations

from bench_loop.suites.agent import AgentSuite
from bench_loop.suites.coding import CodingSuite
from bench_loop.suites.dataextract import DataExtractSuite
from bench_loop.suites.instructfollow import InstructFollowSuite
from bench_loop.suites.reasonmath import ReasonMathSuite
from bench_loop.suites.speed import SpeedSuite
from bench_loop.suites.toolcall import ToolCallSuite

# v2 shipping suites. `coding` runs Python via subprocess sandbox with 10s timeout.
# `tool_use` remains deferred (lower-quality fixtures than `toolcall`).
SUITE_REGISTRY = {
    "speed": SpeedSuite,
    "toolcall": ToolCallSuite,
    "dataextract": DataExtractSuite,
    "instructfollow": InstructFollowSuite,
    "reasonmath": ReasonMathSuite,
    "coding": CodingSuite,
    "agent": AgentSuite,
}

DEFAULT_SUITES = [
    "speed",
    "toolcall",
    "coding",
    "dataextract",
    "instructfollow",
    "reasonmath",
]

__all__ = [
    "AgentSuite",
    "CodingSuite",
    "DataExtractSuite",
    "DEFAULT_SUITES",
    "InstructFollowSuite",
    "ReasonMathSuite",
    "SpeedSuite",
    "SUITE_REGISTRY",
    "ToolCallSuite",
]
