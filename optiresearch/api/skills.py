"""Skill endpoints."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from optiresearch.skills.router import SkillRouter

router = APIRouter(prefix="/v1/skills", tags=["skills"])


class SkillResolveRequest(BaseModel):
    role: str
    task: str
    intent: Optional[str] = None
    budget: Optional[str] = None


@router.post("/resolve")
def resolve_skills(request: SkillResolveRequest) -> list[dict]:
    return [
        skill.model_dump(mode="json")
        for skill in SkillRouter().resolve(
            role=request.role,
            task=request.task,
            intent=request.intent,
            budget=request.budget,
        )
    ]
