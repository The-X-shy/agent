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
    "proxy_as_native_geolens",
    "report_only_as_improvement",
    "structured_unsupported_as_success",
    "lightweight_as_native_physical",
    "synthetic_metric_as_real_hsi",
    "evidence_level_overestimated",
    "handler_capability_exceeded",
    "component_surrogate_as_full_geolens",
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
    # Phase 41: handler-capability-driven ceiling
    final_claim_ceiling: str = ""
    ceiling_source: str = ""
    limiting_factor: str = ""
    downgrade_reasons: list[str] = field(default_factory=list)
    # Phase 46: artifact evidence binding
    evidence_artifact_ids: list[str] = field(default_factory=list)
    evidence_completeness: str = ""
    missing_evidence_artifacts: list[str] = field(default_factory=list)


class ClaimGateV2:
    """Pre-check claims against backend capabilities and known violation patterns."""

    def check_claim(
        self,
        claim_text: str,
        backend_id: str,
        experiment_result: Optional[dict[str, Any]] = None,
        evidence_scope: Optional[dict[str, Any]] = None,
        handler_id: str = "",
    ) -> ClaimGateDecision:
        """Check a proposed claim and return a gating decision.

        Args:
            claim_text: The proposed claim statement.
            backend_id: The backend that produced the evidence.
            experiment_result: Optional result dict with metrics and metadata.
            evidence_scope: Optional scope dict (e.g. local_only, synthetic_data).
            handler_id: Optional handler ID for capability-based ceiling.

        Returns:
            ClaimGateDecision with verdict and safe wording.
        """
        claim_lower = claim_text.lower()
        result = self._normalize_result(experiment_result or {})

        # Phase 41: Resolve claim ceiling from handler capability, not just backend
        resolved = self._resolve_ceiling(backend_id, handler_id, result)
        max_claim = resolved.final_claim_ceiling

        outcome_decision = self._check_plan_execution_outcome(claim_text, claim_lower, result, max_claim)
        if outcome_decision is not None:
            outcome_decision.final_claim_ceiling = resolved.final_claim_ceiling
            outcome_decision.ceiling_source = resolved.ceiling_source
            outcome_decision.limiting_factor = resolved.limiting_factor
            outcome_decision.downgrade_reasons = resolved.downgrade_reasons
            self._publish_claim_event(
                outcome_decision.decision,
                outcome_decision.violation_type,
                claim_text,
            )
            return outcome_decision

        # Detect violations
        violations = self._detect_violations(claim_lower, backend_id, result, evidence_scope or {})

        if not violations:
            self._publish_claim_event("supported", None, claim_text)
            return ClaimGateDecision(
                decision="supported",
                max_allowed_claim=max_claim,
                violation_reason=None,
                violation_type=None,
                safe_wording=claim_text,
                metadata={"outcome": result.get("outcome") or result.get("evidence_level", "")},
                final_claim_ceiling=resolved.final_claim_ceiling,
                ceiling_source=resolved.ceiling_source,
                limiting_factor=resolved.limiting_factor,
                downgrade_reasons=resolved.downgrade_reasons,
            )

        # Use the most severe violation
        primary_type, primary_reason = violations[0]
        decision = self._violation_to_decision(primary_type)
        safe = self._safe_wording(claim_text, primary_type, max_claim)

        self._publish_claim_event(decision, primary_type, claim_text)
        return ClaimGateDecision(
            decision=decision,
            max_allowed_claim=max_claim,
            violation_reason=primary_reason,
            violation_type=primary_type,
            required_additional_evidence=self._required_evidence(primary_type, backend_id),
            safe_wording=safe,
            applicable_caveats=self._caveats(primary_type),
            metadata={"outcome": result.get("outcome") or result.get("evidence_level", "")},
            final_claim_ceiling=resolved.final_claim_ceiling,
            ceiling_source=resolved.ceiling_source,
            limiting_factor=resolved.limiting_factor,
            downgrade_reasons=resolved.downgrade_reasons,
        )

    def _resolve_ceiling(
        self, backend_id: str, handler_id: str, result: dict[str, Any]
    ) -> Any:
        from optiresearch.memory.claim_ceiling_resolver import (
            resolve_claim_ceiling,
        )
        # Only pass constraints that are explicitly present in the result
        has_synthetic = "synthetic_data" in result
        has_physical = "physical_backend" in result
        has_native = "native_backend" in result
        has_fft = "phase_to_fft_proxy_used" in result
        return resolve_claim_ceiling(
            handler_id=handler_id or result.get("handler_id", ""),
            backend_id=backend_id,
            dataset="synthetic" if has_synthetic and result.get("synthetic_data") else ("real" if result.get("real_data") else ""),
            execution_fidelity=result.get("execution_fidelity", ""),
            evidence_level=result.get("evidence_level", ""),
            execution_target=result.get("execution_target", "local"),
            remote_worker_id=result.get("remote_worker_id", ""),
            remote_validation_passed=result.get("remote_validation_passed") if "remote_validation_passed" in result else None,
            synthetic_data=has_synthetic and result.get("synthetic_data", False),
            physical_backend=result.get("physical_backend") if has_physical else None,
            native_backend=result.get("native_backend") if has_native else None,
            real_data=result.get("real_data", False),
            proxy_fallback_used=result.get("proxy_fallback_used", False),
            full_wave_optics=result.get("full_wave_optics", False),
            phase_to_fft_proxy_used=result.get("phase_to_fft_proxy_used") if has_fft else None,
        )

    def _normalize_result(self, result: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(result)
        metrics = normalized.get("metrics")
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                normalized.setdefault(key, value)
        return normalized

    def _check_plan_execution_outcome(
        self,
        claim_text: str,
        claim_lower: str,
        result: dict[str, Any],
        max_claim: str,
    ) -> ClaimGateDecision | None:
        evidence_level = result.get("evidence_level") or result.get("outcome")
        task_type = result.get("task_type", "")
        status = result.get("status", "")

        if evidence_level == "report_only" or task_type == "report_generation":
            if (
                "negative result" in claim_lower
                and ("document" in claim_lower or "report" in claim_lower)
                and not _contains_success_or_improvement_claim(claim_lower)
            ):
                return ClaimGateDecision(
                    decision="supported",
                    max_allowed_claim="report_only",
                    violation_reason=None,
                    violation_type=None,
                    safe_wording=claim_text,
                    applicable_caveats=["Report-only evidence does not support optical improvement"],
                    metadata={"outcome": "report_only"},
                )
            return ClaimGateDecision(
                decision="unsupported",
                max_allowed_claim="report_only",
                violation_reason="Report-only evidence can document a negative result but cannot support optical improvement or task success",
                violation_type="report_only_as_improvement",
                required_additional_evidence=["Run a local scientific experiment that produces measured metrics"],
                safe_wording="The negative result is documented; optical improvement remains unsupported.",
                applicable_caveats=["Report-only evidence"],
                metadata={"outcome": "report_only"},
            )

        if evidence_level == "structured_unsupported" or status == "unsupported":
            boundary_terms = ("boundary", "unsupported", "unavailable", "detected", "limitation")
            if any(term in claim_lower for term in boundary_terms) and not _contains_success_or_improvement_claim(claim_lower):
                return ClaimGateDecision(
                    decision="supported",
                    max_allowed_claim="structured_unsupported",
                    violation_reason=None,
                    violation_type=None,
                    safe_wording=claim_text,
                    applicable_caveats=["Structured unsupported outcome records a boundary, not task success"],
                    metadata={"outcome": "structured_unsupported"},
                )
            return ClaimGateDecision(
                decision="unsupported",
                max_allowed_claim="structured_unsupported",
                violation_reason="Structured unsupported outcome cannot support task success",
                violation_type="structured_unsupported_as_success",
                required_additional_evidence=["Complete a supported local execution path"],
                safe_wording="A local execution boundary was detected; task success is unsupported.",
                applicable_caveats=["Structured unsupported outcome"],
                metadata={"outcome": "structured_unsupported"},
            )

        if evidence_level == "lightweight_scientific_execution" and status in ("completed", "succeeded"):
            return None

        if evidence_level == "local_execution_completed" and status in ("completed", "succeeded"):
            return None

        return None

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

        # 9. proxy_as_native_geolens (Phase 33)
        exp_fidelity = ""
        proxy_used = None
        if result:
            payload = result if isinstance(result, dict) else {}
            exp_fidelity = payload.get("execution_fidelity", "")
            proxy_used = payload.get("phase_to_fft_proxy_used", None)
        if (
            backend_id == "deeplens_geolens_geometric"
            and ("native lens" in claim_lower or "geolens" in claim_lower)
            and (exp_fidelity == "lightweight_proxy" or proxy_used is True)
        ):
            violations.append(
                (
                    "proxy_as_native_geolens",
                    "FFT proxy experiment cannot be claimed as native GeoLens geometric PSF",
                )
            )

        # 10. lightweight_as_native_physical (Phase 39)
        if result.get("evidence_level") == "lightweight_scientific_execution" and (
            "native" in claim_lower
            or "physical" in claim_lower
            or "deep lens" in claim_lower
            or "deeplens" in claim_lower
        ):
            violations.append(
                (
                    "lightweight_as_native_physical",
                    "Lightweight scientific execution uses FFT proxy and synthetic data "
                    "— cannot claim native DeepLens simulation or physical performance",
                )
            )

        # 11. synthetic_metric_as_real_hsi (Phase 39)
        if result.get("synthetic_data") is True and (
            "real hsi" in claim_lower
            or "real measurement" in claim_lower
            or "physical hsi" in claim_lower
        ):
            violations.append(
                (
                    "synthetic_metric_as_real_hsi",
                    "Experimental metrics from synthetic HSI data cannot support "
                    "real HSI or physical measurement claims",
                )
            )

        # 12. evidence_level_overestimated (Phase 40)
        expected_ev = result.get("expected_evidence_level", "")
        actual_ev = result.get("actual_handler_evidence_level") or result.get("evidence_level", "")
        if expected_ev and actual_ev and _evidence_rank(expected_ev) > _evidence_rank(actual_ev):
            violations.append(
                (
                    "evidence_level_overestimated",
                    f"Expected evidence level '{expected_ev}' exceeds actual "
                    f"handler capability '{actual_ev}'",
                )
            )

        # 13. handler_capability_exceeded (Phase 40)
        claim_ceiling = result.get("claim_ceiling", "")
        if claim_ceiling and _evidence_rank_from_claim(claim_lower) > _evidence_rank(claim_ceiling):
            violations.append(
                (
                    "handler_capability_exceeded",
                    f"Claim exceeds handler max ceiling '{claim_ceiling}'",
                )
            )

        # 14. component surrogate cannot be promoted to full GeoLens/lens claims.
        if result.get("evidence_level") == "component_surrogate_hsi_codesign" and (
            "full geolens" in claim_lower
            or "lens-level" in claim_lower
            or "lens level" in claim_lower
            or "native physical lens" in claim_lower
            or "real camera validation" in claim_lower
            or "full wave-optics" in claim_lower
            or "full wave optics" in claim_lower
        ):
            violations.append(
                (
                    "component_surrogate_as_full_geolens",
                    "Component surrogate HSI evidence cannot support full GeoLens, lens-level physical, real camera, or full wave-optics claims",
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
            "proxy_as_native_geolens",
            "black_box_as_native",
            "unsupported_path_as_supported",
            "report_only_as_improvement",
            "structured_unsupported_as_success",
            "lightweight_as_native_physical",
            "synthetic_metric_as_real_hsi",
            "evidence_level_overestimated",
            "handler_capability_exceeded",
            "component_surrogate_as_full_geolens",
        }
        qualified: set[ViolationType] = {
            "differentiable_as_improves",
            "rollback_protection_as_improvement",
            "unsupported_path_as_supported",
            "evidence_level_overestimated",
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
            "improvement": "documented boundary",
            "succeeded": "was attempted",
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
            "report_only_as_improvement": [
                "Run a local scientific experiment with measured metrics",
            ],
            "structured_unsupported_as_success": [
                "Complete a supported local execution path",
            ],
            "lightweight_as_native_physical": [
                "Run native DeepLens GeoLens geometric PSF experiment",
                "Run real HSI measurement with physical camera",
            ],
            "synthetic_metric_as_real_hsi": [
                "Acquire real HSI dataset",
                "Run experiment on real sensor data",
            ],
            "evidence_level_overestimated": [
                "Align expected evidence level with handler capability",
                "Use a handler that can produce the claimed evidence level",
            ],
            "handler_capability_exceeded": [
                "Reduce claim scope to match handler max claim ceiling",
                "Use a higher-capability handler or backend",
            ],
            "component_surrogate_as_full_geolens": [
                "Run a full GeoLens differentiable PSF path with trainable lens parameters",
                "Validate on real HSI/camera data for real performance claims",
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
            "report_only_as_improvement": [
                "Report-only evidence documents a result but does not validate optical performance",
            ],
            "structured_unsupported_as_success": [
                "Unsupported execution records a boundary only",
            ],
            "lightweight_as_native_physical": [
                "Evidence based on synthetic HSI and FFT PSF proxy, not native DeepLens simulation",
            ],
            "synthetic_metric_as_real_hsi": [
                "Evidence based on synthetic HSI data only",
                "Real HSI performance may differ significantly",
            ],
            "evidence_level_overestimated": [
                "Evidence level was downgraded to match handler capability",
            ],
            "handler_capability_exceeded": [
                "This claim exceeds the handler's maximum evidence ceiling",
            ],
            "component_surrogate_as_full_geolens": [
                "Evidence is component-level surrogate HSI only",
                "Full GeoLens, physical lens, real camera, and full wave-optics claims remain unsupported",
            ],
        }
        return caveats.get(violation_type, [])

    def _publish_claim_event(self, decision: str, violation_type: str | None, claim_text: str) -> None:
        try:
            from optiresearch.agent_system.event_bus import get_event_bus
            from optiresearch.agent_system.events import AgentEvent
            event_type = "claim_downgraded" if violation_type else "claim_checked"
            get_event_bus().publish(AgentEvent.create(event_type, "claim_gate",
                payload={"decision": decision, "violation_type": violation_type or "none",
                         "claim_text": claim_text[:200]}))
        except Exception:
            pass


def _contains_success_or_improvement_claim(claim_lower: str) -> bool:
    terms = (
        "improvement",
        "improved",
        "improves",
        "better",
        "succeeded",
        "success",
        "completed",
        "achieved",
    )
    return any(term in claim_lower for term in terms)


def _evidence_rank(level: str) -> int:
    """Return numeric rank for evidence level comparison. Higher = stronger evidence."""
    ranks = {
        "": 0,
        "requires_user_data": 0,
        "structured_unsupported": 0,
        "needs_followup": 0,
        "report_only": 1,
        "negative_result": 1,
        "mock_simulation": 2,
        "deeplens_integration_smoke": 3,
        "native_component_optimization": 4,
        "component_surrogate_hsi_codesign": 5,
        "native_hsi_proxy": 5,
        "native_full_reconstruction_proxy": 6,
        "lightweight_scientific_execution": 7,
        "sweep_analysis": 7,
        "native_lens_simulation": 8,
        "native_waveoptics_simulation": 9,
        "stable_native_lens_hsi_codesign": 10,
        "rollback_protected_native_lens_hsi": 11,
        "real_hsi_performance": 12,
        "real_hsi": 12,
    }
    return ranks.get(level, 0)


def _evidence_rank_from_claim(claim_lower: str) -> int:
    """Estimate evidence rank from claim text content."""
    if any(t in claim_lower for t in ("real hsi", "physical measurement", "production")):
        return 12
    if any(t in claim_lower for t in ("wave-optics", "coherent", "wave optics")):
        return 9
    if any(t in claim_lower for t in ("native lens", "native deeplens", "native simulation")):
        return 8
    if "component surrogate" in claim_lower or "surrogate psf" in claim_lower:
        return 5
    if any(t in claim_lower for t in ("synthetic", "lightweight", "mse-only")):
        return 7
    if "report" in claim_lower or "negative result" in claim_lower:
        return 1
    return 0
