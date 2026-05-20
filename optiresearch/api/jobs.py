"""Run endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field
from fastapi import APIRouter

from optiresearch.runtime.graph import run_mvp_flow

router = APIRouter(prefix="/v1/runs", tags=["runs"])


class MvpRunRequest(BaseModel):
    objective: str = Field(min_length=1)
    workspace_id: str = "default"


@router.post("/mvp")
def run_mvp(request: MvpRunRequest) -> dict:
    return run_mvp_flow(request.objective, workspace_id=request.workspace_id)
