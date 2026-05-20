"""Claim endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from optiresearch.memory.claim_evidence import ClaimEvidenceManager

router = APIRouter(prefix="/v1/claims", tags=["claims"])


@router.get("/{claim_id}")
def get_claim(claim_id: str) -> dict:
    claim = ClaimEvidenceManager().get_claim(claim_id)
    if claim is None:
        raise HTTPException(status_code=404, detail={"error": "claim not found"})
    return claim.model_dump(mode="json")


@router.get("/{claim_id}/explain")
def explain_claim(claim_id: str) -> dict:
    manager = ClaimEvidenceManager()
    if manager.get_claim(claim_id) is None:
        raise HTTPException(status_code=404, detail={"error": "claim not found"})
    return manager.explain_claim(claim_id)
