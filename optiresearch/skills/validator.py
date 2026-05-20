"""Skill result validators."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class SkillValidator:
    def validate_artifacts(self, result: dict[str, Any]) -> list[str]:
        if result.get("status") != "succeeded":
            return []
        errors: list[str] = []
        for path in result.get("artifacts", []):
            if not Path(path).exists():
                errors.append(f"Missing artifact: {path}")
        return errors

    def validate_metrics(self, result: dict[str, Any]) -> list[str]:
        if result.get("status") != "succeeded":
            return []
        metrics = result.get("metrics", {})
        errors: list[str] = []
        for key in (
            "depth_planes",
            "wavelength_bands",
            "psf_depth_similarity",
            "spectral_separability",
            "mock_mtf_mean",
            "mock_energy_efficiency",
        ):
            if key not in metrics:
                errors.append(f"Missing metric: {key}")
        return errors

    def validate_claim_evidence(self, result: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        for claim in result.get("claims", []):
            if claim.get("status") == "supported" and not claim.get("support_edges"):
                errors.append(f"Supported claim lacks evidence: {claim.get('claim_id')}")
        return errors
