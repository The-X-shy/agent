"""Benchmark endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from optiresearch.benchmarks.opti_memory_bench.runner import OptiMemoryBenchRunner

router = APIRouter(prefix="/v1/benchmarks", tags=["benchmarks"])


@router.post("/opti-memory/run")
def run_opti_memory_bench() -> dict:
    return OptiMemoryBenchRunner().run()
