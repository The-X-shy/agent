"""Memory query endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from optiresearch.memory.router import MemoryRouter

router = APIRouter(prefix="/v1/memory", tags=["memory"])


class MemoryQueryRequest(BaseModel):
    role: str = "System"
    intent: str
    query: str
    scope: dict[str, Any] = {}
    top_k: int = 8
    require_evidence: bool = False


@router.post("/query")
def query_memory(request: MemoryQueryRequest) -> dict:
    return MemoryRouter().query(
        role=request.role,
        intent=request.intent,
        query=request.query,
        scope=request.scope,
        top_k=request.top_k,
        require_evidence=request.require_evidence,
    )
