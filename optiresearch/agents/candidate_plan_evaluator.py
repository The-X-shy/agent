"""Candidate Plan Evaluator for Phase 36.

Scores and ranks ExperimentDesignCandidates across 8 dimensions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from optiresearch.agents.experiment_design_generator import ExperimentDesignCandidate


@dataclass
class PlanScore:
    design_id: str
    total_score: float
    score_breakdown: dict[str, float] = field(default_factory=dict)
    recommendation: str = ""
    reason: str = ""
    # Phase 56: diagnosis-aware scoring metadata
    diagnosis_score_bonus: float = 0.0
    diagnosis_factors_used: list[str] = field(default_factory=list)
    selected_due_to_diagnosis: bool = False
    scoring_explanation: str = ""


@dataclass
class ExecutableDesignSelection:
    mode: str
    selected_designs: list[ExperimentDesignCandidate] = field(default_factory=list)
    selected_design: str | None = None
    selected_design_rank: int | None = None
    skipped_higher_ranked_designs: list[dict[str, Any]] = field(default_factory=list)
    executable_selection_reason: str = ""
    stop_reason: str | None = None


class CandidatePlanEvaluator:
    def __init__(self):
        self._weights = {
            "evidence_gain_score": 0.20,
            "metric_gain_likelihood": 0.15,
            "execution_feasibility": 0.15,
            "backend_availability": 0.15,
            "runtime_cost": 0.10,
            "claim_safety": 0.10,
            "novelty": 0.05,
            "risk_penalty": 0.10,
        }
        self._diagnosis_context: dict[str, Any] = {}

    def set_diagnosis_context(self, diagnosis: dict[str, Any]) -> None:
        self._diagnosis_context = diagnosis

    def evaluate(self, designs: list[ExperimentDesignCandidate]) -> list[PlanScore]:
        scores = []
        for d in designs:
            breakdown = self._score_design(d)
            # Phase 56: Apply diagnosis-aware bonus
            diag_bonus = self._apply_diagnosis_bonus(d)
            breakdown["diagnosis_score_bonus"] = diag_bonus
            total = sum(breakdown.values())
            rec, reason = self._recommend(total, d)
            diag_factors = self._diagnosis_factors_used(d)
            scores.append(PlanScore(
                design_id=d.design_id,
                total_score=round(total, 3),
                score_breakdown=breakdown,
                recommendation=rec,
                reason=reason,
                diagnosis_score_bonus=diag_bonus,
                diagnosis_factors_used=diag_factors,
                scoring_explanation=f"Diagnosis bonus: {diag_bonus:.3f}" if diag_bonus > 0 else "",
            ))
        scores.sort(key=lambda s: s.total_score, reverse=True)
        return scores

    def _apply_diagnosis_bonus(self, d: ExperimentDesignCandidate) -> float:
        """Apply scoring bonus based on diagnosis failure modes matching DeepLens diagnostic designs."""
        ctx = self._diagnosis_context
        if not ctx or ctx.get("status") != "diagnosed":
            return 0.0

        failure_modes = ctx.get("failure_modes", [])
        likely_causes = ctx.get("likely_causes", [])
        bonus = 0.0
        did = d.design_id
        family = getattr(d, "strategy_family", "")

        # no_parameter_change → boost autograd audit, parameter inspection, surface freeze
        if "no_parameter_change" in failure_modes:
            if any(kw in did for kw in ("autograd_audit", "trainable_parameter", "surface_freeze_unfreeze")):
                bonus += 0.15
            if "verify_trainable" in did:
                bonus += 0.15

        # unstable_training → boost curriculum, regularization, component-first
        if "unstable_training" in failure_modes:
            if any(kw in did for kw in ("curriculum_probe", "regularized_probe", "component_first_fresnel")):
                bonus += 0.12
            if family in ("curriculum_learning", "optical_regularization", "component_first"):
                bonus += 0.08

        # gradient_flow_blocked → strong boost for autograd audit, component-first
        if "gradient_flow_blocked" in likely_causes:
            if any(kw in did for kw in ("autograd_audit", "autograd_graph_audit")):
                bonus += 0.18
            if any(kw in did for kw in ("component_first", "component_level_geolens")):
                bonus += 0.12

        # Diagnostic information gain: probe_only designs get extra weight
        if getattr(d, "probe_only", False):
            bonus += 0.05
        if d.expected_evidence_level == "diagnostic_evidence":
            bonus += 0.03
        recoveries = ctx.get("recommended_recoveries", [])
        if "component_surrogate" in did and any(
            isinstance(r, str) and r.startswith("component_probe_succeeded:")
            for r in recoveries
        ):
            bonus += 0.20
        if "full_geolens_direct_update_blocked" in failure_modes and "component_surrogate" in did:
            bonus += 0.10
        geolens_audit_connected = (
            ctx.get("graph_connected") is True
            and ctx.get("psf_requires_grad") is True
            and ctx.get("loss_requires_grad") is True
            and int(ctx.get("trainable_param_count") or ctx.get("parameter_count") or 0) > 0
            and int(ctx.get("params_with_grad") or 0) > 0
        )
        if geolens_audit_connected and any(
            kw in did for kw in ("full_geolens_geometric_direct_update", "remote_native_geolens_validation")
        ):
            bonus += 0.20

        return round(bonus, 3)

    def _diagnosis_factors_used(self, d: ExperimentDesignCandidate) -> list[str]:
        ctx = self._diagnosis_context
        if not ctx:
            return []
        factors: list[str] = []
        did = d.design_id
        fm = ctx.get("failure_modes", [])
        lc = ctx.get("likely_causes", [])
        if "no_parameter_change" in fm and any(kw in did for kw in ("autograd", "trainable", "freeze")):
            factors.append("no_parameter_change")
        if "unstable_training" in fm and any(kw in did for kw in ("curriculum", "regularized", "component_first_fresnel")):
            factors.append("unstable_training")
        if "gradient_flow_blocked" in lc and "autograd" in did:
            factors.append("gradient_flow_blocked")
        if "component_surrogate" in did:
            if any(isinstance(r, str) and r.startswith("component_probe_succeeded:") for r in ctx.get("recommended_recoveries", [])):
                factors.append("component_probe_succeeded")
            if "full_geolens_direct_update_blocked" in fm:
                factors.append("full_geolens_direct_update_blocked")
        geolens_audit_connected = (
            ctx.get("graph_connected") is True
            and ctx.get("psf_requires_grad") is True
            and ctx.get("loss_requires_grad") is True
            and int(ctx.get("trainable_param_count") or ctx.get("parameter_count") or 0) > 0
            and int(ctx.get("params_with_grad") or 0) > 0
        )
        if geolens_audit_connected and any(
            kw in did for kw in ("full_geolens_geometric_direct_update", "remote_native_geolens_validation")
        ):
            factors.append("geolens_native_autograd_connected")
        if getattr(d, "probe_only", False):
            factors.append("probe_only")
        if d.expected_evidence_level == "diagnostic_evidence":
            factors.append("diagnostic_evidence")
        return factors

    def select_executable_designs(
        self,
        plan_scores: list[PlanScore],
        designs: list[ExperimentDesignCandidate],
        mode: str,
        limit: int = 1,
        allow_remote: bool = False,
    ) -> ExecutableDesignSelection:
        """Select designs that can be executed under the requested mode."""
        limit = max(1, limit)
        by_id = {d.design_id: d for d in designs}
        ranked: list[tuple[int, PlanScore, ExperimentDesignCandidate]] = []
        for rank, score in enumerate(plan_scores, 1):
            design = by_id.get(score.design_id)
            if design is not None:
                ranked.append((rank, score, design))

        if mode == "dry_run":
            chosen = [d for _, _, d in ranked[:limit]]
            return ExecutableDesignSelection(
                mode=mode,
                selected_designs=chosen,
                selected_design=chosen[0].design_id if chosen else None,
                selected_design_rank=ranked[0][0] if chosen else None,
                executable_selection_reason=(
                    "Dry run keeps the top-ranked design for display; no local execution is attempted."
                    if chosen else "No candidate design available for dry run."
                ),
                stop_reason=None if chosen else "no_candidate_design",
            )

        selected: list[ExperimentDesignCandidate] = []
        skipped: list[dict[str, Any]] = []
        first_selected_rank: int | None = None
        if mode == "remote_opt_in" and allow_remote:
            remote_ranked = [
                (rank, score, design)
                for rank, score, design in ranked
                if _is_remote_required_design(design)
            ]
            ranked = remote_ranked + [
                item for item in ranked if item not in remote_ranked
            ]

        deferred_report_designs: list[tuple[int, PlanScore, ExperimentDesignCandidate]] = []
        for rank, score, design in ranked:
            reason = self._non_executable_reason(score, design, mode, allow_remote)
            if reason:
                if first_selected_rank is None:
                    skipped.append({
                        "design_id": design.design_id,
                        "rank": rank,
                        "recommendation": score.recommendation,
                        "skipped_reason": reason,
                    })
                continue
            if mode == "local" and _is_report_only_design(design):
                deferred_report_designs.append((rank, score, design))
                if first_selected_rank is None:
                    skipped.append({
                        "design_id": design.design_id,
                        "rank": rank,
                        "recommendation": score.recommendation,
                        "skipped_reason": "reserved_report_fallback",
                    })
                continue
            if first_selected_rank is None:
                first_selected_rank = rank
            selected.append(design)
            if len(selected) >= limit:
                break

        if not selected and deferred_report_designs:
            for rank, _score, design in deferred_report_designs:
                if first_selected_rank is None:
                    first_selected_rank = rank
                selected.append(design)
                if len(selected) >= limit:
                    break
            skipped = [
                item for item in skipped
                if item.get("skipped_reason") != "reserved_report_fallback"
            ]

        if not selected:
            if not skipped:
                for rank, score, design in ranked:
                    skipped.append({
                        "design_id": design.design_id,
                        "rank": rank,
                        "recommendation": score.recommendation,
                        "skipped_reason": self._non_executable_reason(score, design, mode, allow_remote) or "not_selected",
                    })
            return ExecutableDesignSelection(
                mode=mode,
                selected_designs=[],
                skipped_higher_ranked_designs=skipped,
                executable_selection_reason="No executable design found for the requested mode.",
                stop_reason="no_executable_design",
            )

        return ExecutableDesignSelection(
            mode=mode,
            selected_designs=selected,
            selected_design=selected[0].design_id,
            selected_design_rank=first_selected_rank,
            skipped_higher_ranked_designs=skipped,
            executable_selection_reason=(
                f"Selected highest-ranked local executable design: {selected[0].design_id}."
                if mode == "local"
                else f"Selected highest-ranked executable design for {mode}: {selected[0].design_id}."
            ),
        )

    def _score_design(self, d: ExperimentDesignCandidate) -> dict[str, float]:
        evidence_map = {"negative_result": 0.3, "lightweight_scientific_execution": 0.4,
                        "component_surrogate_hsi_codesign": 0.55,
                        "native_lens_simulation": 0.5,
                        "native_waveoptics_simulation": 0.8, "real_hsi": 1.0,
                        "sweep_analysis": 0.4, "requires_user_data": 0.0,
                        "structured_unsupported": 0.1, "needs_followup": 0.1,
                        "report_only": 0.2, "": 0.1}
        # Use actual handler evidence level if available, fall back to expected
        effective_evidence = d.actual_handler_evidence_level or d.expected_evidence_level
        evidence_score = evidence_map.get(effective_evidence, 0.3)
        # Penalty for evidence downgrade
        if d.evidence_alignment_status == "downgraded_to_handler_capability":
            evidence_score = evidence_map.get(d.actual_handler_evidence_level, 0.3) - 0.05
        metric_likelihood = 0.3 if "gradient_instability" in d.expected_failure_modes else 0.6
        if d.expected_evidence_level == "negative_result":
            metric_likelihood = 0.1
        elif effective_evidence == "lightweight_scientific_execution":
            metric_likelihood = 0.9
        elif effective_evidence in ("structured_unsupported", "needs_followup", "requires_user_data"):
            metric_likelihood = 0.0
        risk_map = {"low": 1.0, "medium": 0.7, "high": 0.4}
        rt_map = {0: 1.0, 60: 1.0, 600: 0.8, 3600: 0.5, 7200: 0.3}
        # Execution feasibility lowered for non-local handlers
        exec_feasibility = risk_map.get(d.risk_level, 0.7)
        if d.actual_handler_evidence_level in ("structured_unsupported", "needs_followup"):
            exec_feasibility = 0.3
        elif d.actual_handler_evidence_level == "requires_user_data":
            exec_feasibility = 0.0
        return {
            "evidence_gain_score": evidence_score * self._weights["evidence_gain_score"],
            "metric_gain_likelihood": metric_likelihood * self._weights["metric_gain_likelihood"],
            "execution_feasibility": exec_feasibility * self._weights["execution_feasibility"],
            "backend_availability": (0.8 if d.backend_id else 1.0) * self._weights["backend_availability"],
            "runtime_cost": rt_map.get(d.estimated_runtime_sec, 0.5) * self._weights["runtime_cost"],
            "claim_safety": (0.9 if d.claim_ceiling else 0.5) * self._weights["claim_safety"],
            "novelty": (0.8 if "waveoptics" in d.design_id or "real" in d.design_id else 0.4) * self._weights["novelty"],
            "risk_penalty": -risk_map.get(d.risk_level, 0.7) * self._weights["risk_penalty"],
        }

    def _recommend(self, total: float, d: ExperimentDesignCandidate) -> tuple[str, str]:
        if d.estimated_runtime_sec == 0:
            return "needs_user_data", "Requires external data — cannot execute autonomously"
        if total >= 0.6:
            return "execute_now", "High score with acceptable risk"
        if total >= 0.45:
            return "dry_run_first", "Medium score — validate preconditions before execution"
        if "remote" in d.design_id or d.estimated_runtime_sec > 1800:
            return "needs_remote", "Requires remote execution or long runtime"
        if total < 0.3:
            return "defer", "Low score — revisit after gathering more evidence"
        return "execute_now", "Acceptable score"

    def _non_executable_reason(
        self,
        score: PlanScore,
        design: ExperimentDesignCandidate,
        mode: str,
        allow_remote: bool,
    ) -> str | None:
        rec = score.recommendation
        if rec == "needs_user_data" or design.estimated_runtime_sec == 0 or design.spec_payload.get("action") == "request_real_data":
            return "needs_user_data"
        # Remote-aware routing
        handler_id = getattr(design, "handler_id", "")
        cap = _get_handler_capability(handler_id) if handler_id else None
        if cap:
            if cap.remote_required:
                if mode == "remote_opt_in" and allow_remote:
                    return None  # Selectable in remote mode
                return "needs_remote"
            # Non-remote handlers are fine in remote_opt_in — they run locally
            if mode == "remote_opt_in" and allow_remote:
                return None  # Either remote-capable or locally executable
        if rec == "needs_remote" and not allow_remote:
            return "needs_remote"
        if mode == "remote_opt_in" and rec == "needs_remote" and allow_remote:
            return None
        if mode == "local":
            if cap and cap.remote_required:
                return "needs_remote"
            if rec not in ("execute_now", "dry_run_first"):
                return rec or "not_recommended"
            if not _is_local_supported_design(design):
                return "unsupported_backend"
        return None

    def export(self, scores: list[PlanScore],
               output_path: str | Path | None = None) -> Path:
        path = Path(output_path or "workspace/reports/candidate_plan_scores.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [{"design_id": s.design_id, "total_score": s.total_score,
                 "score_breakdown": s.score_breakdown,
                 "recommendation": s.recommendation, "reason": s.reason}
                for s in scores]
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def export_markdown(self, scores: list[PlanScore],
                        output_path: str | Path | None = None) -> Path:
        path = Path(output_path or "workspace/reports/candidate_plan_scores.md")
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Candidate Plan Scores",
            "",
            "| Rank | Design | Score | Recommendation | Reason |",
            "|---|---|---|---|---|",
        ]
        for i, s in enumerate(scores, 1):
            lines.append(
                f"| {i} | {s.design_id} | {s.total_score:.3f} | "
                f"{s.recommendation} | {s.reason} |"
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path


def _is_local_supported_design(design: ExperimentDesignCandidate) -> bool:
    if _is_report_only_design(design):
        return True
    if design.design_id == "backend_switch_waveoptics_coherent":
        return True
    if "component_surrogate" in design.design_id or design.task_type == "component_surrogate_hsi_codesign":
        return True
    if _is_lightweight_scientific_design(design):
        return True
    if not design.backend_id:
        return False
    try:
        from optiresearch.backends.registry import get_backend, get_backend_task_evidence_cap
        if get_backend(design.backend_id) is None:
            return False
        return get_backend_task_evidence_cap(design.backend_id, design.task_type) is not None
    except Exception:
        return False


def _get_handler_capability(handler_id: str) -> Any:
    try:
        from optiresearch.skills.handler_capability_registry import (
            get_handler_capability_registry,
        )
        return get_handler_capability_registry().get(handler_id)
    except Exception:
        return None


def _is_remote_required_design(design: ExperimentDesignCandidate) -> bool:
    handler_id = getattr(design, "handler_id", "")
    cap = _get_handler_capability(handler_id) if handler_id else None
    return bool(cap and cap.remote_required)


def _is_report_only_design(design: ExperimentDesignCandidate) -> bool:
    return (
        design.required_skills == ["report_generation"]
        or design.spec_payload.get("action") == "export_system_subunit_report"
        or design.design_id == "report_negative_result_doc"
    )


def _is_lightweight_scientific_design(design: ExperimentDesignCandidate) -> bool:
    """Check if this design can be handled by a lightweight scientific handler."""
    if design.design_id in ("objective_redesign_simpler_metric_mse_only", "param_reduction_sweep"):
        return True
    if "component_surrogate" in design.design_id or design.task_type == "component_surrogate_hsi_codesign":
        return True
    if design.spec_payload.get("param_subset"):
        return True
    loss_weights = design.spec_payload.get("loss_weights", {})
    if isinstance(loss_weights, dict) and loss_weights.get("mse", 0) == 1.0:
        if loss_weights.get("spectral_angle", 0) == 0.0:
            return True
    return False
