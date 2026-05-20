"""Trace endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from optiresearch.memory.meta_trace import MetaTraceWriter
from optiresearch.memory.schemas import MetaTrace

router = APIRouter(prefix="/v1/traces", tags=["traces"])


@router.post("")
def write_trace(trace: MetaTrace) -> dict:
    written = MetaTraceWriter().write_trace(trace)
    return written.model_dump(mode="json")


@router.get("/{trace_id}")
def get_trace(trace_id: str) -> dict:
    trace = MetaTraceWriter().get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail={"error": "trace not found"})
    return trace.model_dump(mode="json")
