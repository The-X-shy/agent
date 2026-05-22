"""Claim Gate v2 — pre-check claims before they are written.

Runs before ClaimEvidenceManager to catch violations early.
Detects 8 violation types and enforces backend claim ceilings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

GateDecision = Literal["supported", "qualified", "needs_followup", "unsupported"]

ViolationType = Literal[
    "proxy_as_waveoptics",
    "geometric_as_coherent",
    "synthetic_as_real",
    "differentiable_as_improves",
    "local_only_as_robust",
    "rollback_protection_as_improvement",
    "unsupported_path_as_supported",
    "black_box_as_native",
]


@dataclass
class ClaimGateDecision:
    """Output of claim gate pre-check."""

    decision: GateDecision
    max_allowed_claim: Optional[str]
    violation_reason: Optional[str]
    violation_type: Optional[ViolationType]
    required_additional_evidence: list[str] = field(default_factory=list)
    safe_wording: str = ""
    applicable_caveats: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class ClaimGateV2:
    """Pre-check claims against backend capabilities and known violation patterns."""

    def check_claim(
        self,
        claim_text: str,
        backend_id: str,
        experiment_result: Optional[dict[str, Any]] = None,
        evidence_scope: Optional[dict[str, Any]] = None,
    ) -> ClaimGateDecision:
        """Check a proposed claim and return a gating decision.

        Args:
            claim_text: The proposed claim statement.
            backend_id: The backend that produced the evidence.
            experiment_result: Optional result dict with metrics and metadata.
            evidence_scope: Optional scope dict (e.g. local_only, synthetic_data).

        Returns:
            ClaimGateDecision with verdict and safe wording.
        """
        claim_lower = claim_text.lower()
        result = experiment_result or {}

        # Get backend claim ceiling
        max_claim = self._compute_max_allowed_claim(backend_id)

        # Detect violations
        violations = self._detect_violations(claim_lower, backend_id, result, evidence_scope or {})

        if not violations:
            return ClaimGateDecision(
                decision="supported",
                max_allowed_claim=max_claim,
                violation_reason=None,
                violation_type=None,
                safe_wording=claim_text,
            )

        # Use the most severe violation
        primary_type, primary_reason = violations[0]
        decision = self._violation_to_decision(primary_type)
        safe = self._safe_wording(claim_text, primary_type, max_claim)

        return ClaimGateDecision(
            decision=decision,
            max_allowed_claim=max_claim,
            violation_reason=primary_reason,
            violation_type=primary_type,
            required_additional_evidence=self._required_evidence(primary_type, backend_id),
            safe_wording=safe,
            applicable_caveats=self._caveats(primary_type),
        )

    def _detect_violations(
        self,
        claim_lower: str,
        backend_id: str,
        result: dict[str, Any],
        scope: dict[str, Any],
    ) -> list[tuple[ViolationType, str]]:
        """Detect all applicable violations, ordered by severity."""
        violations: list[tuple[ViolationType, str]] = []

        # 1. proxy_as_waveoptics
        if backend_id in ("phase_to_fft_proxy",) and (
            "wave" in claim_lower or "coherent" in claim_lower
        ):
            violations.append(
                (
                    "proxy_as_waveoptics",
                    "phase-to-FFT proxy cannot claim native wave-optics or coherent behaviour",
                )
            )

        # 2. geometric_as_coherent
        if backend_id == "deeplens_geolens_geometric" and (
            "coherent" in claim_lower
            or "wave-optics" in claim_lower
            or "wave optics" in claim_lower
            or ("full" in claim_lower and "wave" in claim_lower)
        ):
            violations.append(
                (
                    "geometric_as_coherent",
                    "GeoLens geometric PSF cannot claim coherent wave-optics behaviour",
                )
            )

        # 3. synthetic_as_real
        if backend_id in ("local_synthetic_hsi", "mock_deeplens") and (
            "real" in claim_lower or "physical" in claim_lower
        ):
            violations.append(
                (
                    "synthetic_as_real",
                    "Synthetic/mock data cannot support real-world or physical performance claims",
                )
            )

        # 4. differentiable_as_improves
        if (
            "differentiable" in claim_lower
            and ("improves" in claim_lower or "better" in claim_lower)
            and result.get("reconstruction_loss_after", 0) >= result.get(
                "reconstruction_loss_before", 0
            )
        ):
            violations.append(
                (
                    "differentiable_as_improves",
                    "Differentiability does not imply performance improvement — "
                    "loss did not decrease",
                )
            )

        # 5. local_only_as_robust
        if scope.get("execution_target") == "local" and (
            "robust" in claim_lower or "production" in claim_lower
        ):
            violations.append(
                (
                    "local_only_as_robust",
                    "Local-only execution cannot support robustness or production claims "
                    "without remote validation",
                )
            )

        # 6. rollback_protection_as_improvement
        if "rollback" in claim_lower and "improve" in claim_lower:
            accepted = result.get("accepted_update_count", 0)
            if accepted == 0:
                violations.append(
                    (
                        "rollback_protection_as_improvement",
                        "Rollback protected against harmful updates but no updates were "
                        "accepted — this is NOT optical improvement",
                    )
                )

        # 7. unsupported_path_as_supported
        if backend_id == "deeplens_coherent_asm" and "supported" in claim_lower:
            violations.append(
                (
                    "unsupported_path_as_supported",
                    "Coherent ASM path has requires_grad=False — cannot be claimed as supported",
                )
            )

        # 8. black_box_as_native
        if backend_id in ("deeplens_blackbox_source_psf",) and (
            "native gradient" in claim_lower or "native optimization" in claim_lower
        ):
            violations.append(
                (
                    "black_box_as_native",
                    "Black-box backend cannot claim native gradient or native optimization",
                )
            )

        return violations

    def _compute_max_allowed_claim(self, backend_id: str) -> str:
        """Query the backend registry for claim ceiling."""
        try:
            from optiresearch.backends.registry import get_backend

            backend = get_backend(backend_id)
            if backend is not None:
                return backend.claim_ceiling
        except Exception:
            pass
        return "unsupported"

    def _violation_to_decision(self, violation_type: ViolationType) -> GateDecision:
        """Map violation type to gate decision."""
        fatal: set[ViolationType] = {
            "proxy_as_waveoptics",
            "geometric_as_coherent",
            "synthetic_as_real",
            "black_box_as_native",
            "unsupported_path_as_supported",
        }
        qualified: set[ViolationType] = {
            "differentiable_as_improves",
            "rollback_protection_as_improvement",
            "unsupported_path_as_supported",
        }
        if violation_type in fatal:
            return "unsupported"
        if violation_type in qualified:
            return "qualified"
        return "needs_followup"

    def _safe_wording(
        self,
        original: str,
        violation_type: ViolationType,
        max_claim: str,
    ) -> str:
        """Generate safe claim wording based on violation type and claim ceiling."""
        replacements = {
            "wave-optics": "geometric ray-tracing",
            "coherent": "geometric",
            "real": "synthetic",
            "physical": "simulated",
            "full wave-optics": "geometric differentiable",
            "native optimization": "native simulation (claim ceiling: {})".format(max_claim),
            "robust execution": "local execution (remote validation pending)",
            "optical improvement": "rollback-protected training",
            "supported": "partially supported (see caveats)",
        }
        safe = original
        for pattern, replacement in replacements.items():
            if pattern in safe.lower():
                safe = safe.replace(pattern, replacement)
                safe = safe.replace(pattern.title(), replacement)
        if max_claim and max_claim != "unsupported":
            safe += f" [evidence ceiling: {max_claim}]"
        return safe

    def _required_evidence(
        self,
        violation_type: ViolationType,
        backend_id: str,
    ) -> list[str]:
        """List additional evidence needed to resolve the violation."""
        evidence: dict[ViolationType, list[str]] = {
            "proxy_as_waveoptics": [
                "Run native wave-optics probe on DeepLens coherent ASM path",
                "Verify requires_grad=True on coherent PSF output",
            ],
            "geometric_as_coherent": [
                "Switch to DiffractiveLens or pure wave propagation backend",
                "Verify coherent wave-optics gradient flow",
            ],
            "synthetic_as_real": [
                "Acquire real HSI dataset",
                "Run experiment on real sensor data",
            ],
            "differentiable_as_improves": [
                "Reduce optical learning rate",
                "Enable rollback protection",
                "Show loss decreased after optical update",
            ],
            "local_only_as_robust": [
                "Run remote validation on WSL DeepLens instance",
                "Show consistent results across local and remote",
            ],
            "rollback_protection_as_improvement": [
                "Achieve at least one accepted optical update",
                "Show loss decreased after accepted update",
            ],
            "unsupported_path_as_supported": [
                "Identify alternative differentiable path",
                "Verify autograd chain end-to-end",
            ],
            "black_box_as_native": [
                "Use differentiable backend (phase_to_fft_proxy or geolens_geometric)",
                "Verify requires_grad on PSF output",
            ],
        }
        return evidence.get(violation_type, [])

    def _caveats(self, violation_type: ViolationType) -> list[str]:
        """Standard caveats for each violation type."""
        caveats: dict[ViolationType, list[str]] = {
            "proxy_as_waveoptics": [
                "Evidence based on scalar phase FFT proxy, not full wave propagation",
            ],
            "geometric_as_coherent": [
                "Evidence based on geometric ray-tracing, not coherent wave-optics",
            ],
            "synthetic_as_real": [
                "Evidence based on synthetic data only",
                "Real-world performance may differ",
            ],
            "differentiable_as_improves": [
                "Differentiability is a capability, not a performance guarantee",
            ],
            "local_only_as_robust": [
                "Local execution only — remote validation pending",
            ],
            "rollback_protection_as_improvement": [
                "Rollback prevents regression but does not demonstrate improvement",
            ],
            "unsupported_path_as_supported": [
                "This backend path is not currently differentiable",
            ],
            "black_box_as_native": [
                "This backend does not expose gradient information",
            ],
        }
        return caveats.get(violation_type, [])
