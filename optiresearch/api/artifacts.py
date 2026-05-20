"""Artifact endpoints."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter

from optiresearch.storage.file_artifact_store import FileArtifactStore

router = APIRouter(prefix="/v1/artifacts", tags=["artifacts"])


@router.get("")
def list_artifacts(run_id: Optional[str] = None) -> list[dict]:
    return [artifact.model_dump(mode="json") for artifact in FileArtifactStore().list_artifacts(run_id=run_id)]
