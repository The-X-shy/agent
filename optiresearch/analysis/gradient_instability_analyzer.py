"""Gradient Instability Analyzer for Phase 51."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from optiresearch.memory.schemas import make_deterministic_id
from optiresearch.schemas.gradient_instability import (
    GradientInstabilityDiagnosis,
    GradientInstabilityMetrics,
)


def analyze_gradient_instability(
    source_paths: list[str] | None = None,
    remote_job_ids: list[str] | None = None,
) -> GradientInstabilityDiagnosis:
    """Analyze GeoLens gradient instability from sweep results and remote job data."""
    sources: list[dict[str, Any]] = []
    source_files: list[str] = []

    for path in (source_paths or []):
        p = Path(path)
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                sources.append(data)
                source_files.append(str(p))
            except Exception:
                pass

    for jid in (remote_job_ids or []):
        jp = Path(f"workspace/remote_jobs/{jid}/command_result.json")
        if jp.exists():
            try:
                data = json.loads(jp.read_text(encoding="utf-8"))
                sources.append(data)
                source_files.append(str(jp))
            except Exception:
                pass
        mp = Path(f"workspace/remote_jobs/{jid}/metrics_summary.json")
        if mp.exists():
            try:
                data = json.loads(mp.read_text(encoding="utf-8"))
                sources.append(data)
                source_files.append(str(mp))
            except Exception:
                pass

    diagnosis_id = make_deterministic_id("gdiag", str(source_files), "v1")
    diagnosis = GradientInstabilityDiagnosis(
        diagnosis_id=diagnosis_id,
        source_paths=source_files,
        source_count=len(sources),
    )

    if not sources:
        diagnosis.status = "insufficient_evidence"
        diagnosis.warnings.append("No source data found")
        return diagnosis

    metrics = _extract_metrics(sources)
    diagnosis.metrics = metrics
    diagnosis.failure_modes = _classify_failure_modes(metrics)
    diagnosis.severity = _assess_severity(metrics, diagnosis.failure_modes)
    diagnosis.likely_causes = _infer_likely_causes(metrics, diagnosis.failure_modes)
    diagnosis.recommended_recoveries = _recommend_recoveries(diagnosis.failure_modes)
    diagnosis.claim_implications = _claim_implications(diagnosis.failure_modes)
    diagnosis.next_experiment_design_hints = _design_hints(diagnosis.failure_modes, diagnosis.likely_causes)

    if diagnosis.failure_modes:
        diagnosis.status = "diagnosed"

    return diagnosis


def _ingest_remote_diagnostic_metrics(sources: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract remote diagnostic metrics from source data."""
    diag: dict[str, Any] = {}
    for s in sources:
        for key in ("trainable_param_count", "params_with_grad", "graph_connected",
                     "candidate_update_changes_parameter", "psf_requires_grad",
                     "loss_requires_grad", "detach_suspected", "grad_norm_max",
                     "grad_norm_mean", "parameter_count", "trainable_count",
                     "resolved_lens_file", "lens_resolution_source"):
            if key in s:
                diag[key] = s[key]
    return diag


def _extract_metrics(sources: list[dict[str, Any]]) -> GradientInstabilityMetrics:
    configs: list[dict[str, Any]] = []
    for s in sources:
        if "configs" in s:
            configs.extend(s["configs"])
        if "result_payload" in s:
            configs.append(s["result_payload"])
    if not configs:
        configs = sources

    grad_max = None
    grad_mean = None
    recon_grad = None
    total_accepted = 0
    total_rejected = 0
    total_rollback = 0
    loss_before = None
    loss_after = None
    stable = False
    params_changed = False
    ev_level = ""
    fidelity = ""
    proxy = False

    for c in configs:
        gm = c.get("optical_gradient_norm_max", c.get("optical_gradient_norm"))
        if gm is not None:
            grad_max = max(grad_max or 0, gm)
        gmean = c.get("optical_gradient_norm_mean")
        if gmean is not None:
            grad_mean = gmean
        rg = c.get("recon_gradient_norm")
        if rg is not None:
            recon_grad = rg
        total_accepted += c.get("accepted_update_count", 0)
        total_rejected += c.get("rejected_update_count", 0)
        total_rollback += c.get("rollback_count", 0)
        lb = c.get("reconstruction_loss_before", c.get("loss_before"))
        la = c.get("reconstruction_loss_after", c.get("loss_after"))
        if lb is not None and (loss_before is None or lb < loss_before):
            loss_before = lb
        if la is not None and (loss_after is None or la < loss_after):
            loss_after = la
        if c.get("stable_training_succeeded"):
            stable = True
        if c.get("optical_parameters_changed") or c.get("accepted_update_count", 0) > 0:
            params_changed = True
        if c.get("evidence_level"):
            ev_level = c["evidence_level"]
        if c.get("execution_fidelity"):
            fidelity = c["execution_fidelity"]
        if c.get("proxy_fallback_used"):
            proxy = True

    # Phase 59: Remote diagnostic metrics
    remote_diag = _ingest_remote_diagnostic_metrics(sources)
    if remote_diag:
        if remote_diag.get("trainable_param_count", 0) > 0 and remote_diag.get("params_with_grad", 0) == 0:
            if not params_changed:
                params_changed = False
        if remote_diag.get("graph_connected") is True and not stable:
            pass

    total_updates = total_accepted + total_rejected
    rollback_rate = total_rollback / max(total_updates, 1)

    return GradientInstabilityMetrics(
        optical_gradient_norm_max=grad_max,
        optical_gradient_norm_mean=grad_mean,
        recon_gradient_norm=recon_grad,
        accepted_update_count=total_accepted,
        rejected_update_count=total_rejected,
        rollback_count=total_rollback,
        rollback_rate=rollback_rate,
        reconstruction_loss_before=loss_before,
        reconstruction_loss_after=loss_after,
        loss_delta=(loss_after - loss_before) if loss_before is not None and loss_after is not None else None,
        stable_training_succeeded=stable,
        optical_parameters_changed=params_changed,
        evidence_level=ev_level,
        execution_fidelity=fidelity,
        proxy_fallback_used=proxy,
    )


def _classify_failure_modes(m: GradientInstabilityMetrics) -> list[str]:
    modes: list[str] = []
    if m.optical_gradient_norm_max is not None and m.optical_gradient_norm_max > 1000:
        modes.append("excessive_gradient_norm")
    if m.optical_gradient_norm_max is not None and m.optical_gradient_norm_max > 5000:
        modes.append("extreme_gradient_spike")
    if m.accepted_update_count == 0 and m.rollback_count > 0:
        modes.append("all_updates_rollback")
    if not m.optical_parameters_changed:
        modes.append("no_parameter_change")
    if not m.stable_training_succeeded:
        modes.append("unstable_training")
    if m.loss_delta is not None and m.loss_delta >= 0:
        modes.append("loss_not_improved")
    if not modes:
        modes.append("insufficient_metrics")
    return modes


def _classify_remote_diagnostic_failure_modes(
    trainable_param_count: int,
    params_with_grad: int,
    graph_connected: bool,
    candidate_update_changes_parameter: bool,
) -> list[str]:
    """Classify failure modes from remote diagnostic metrics."""
    modes: list[str] = []
    if trainable_param_count > 0 and params_with_grad == 0:
        modes.append("gradient_flow_blocked")
    if params_with_grad > 0 and not candidate_update_changes_parameter:
        modes.append("optimizer_update_blocked")
    if graph_connected and not candidate_update_changes_parameter:
        modes.append("objective_or_update_instability")
    return modes


def _recommend_from_remote_diagnostics(failure_modes: list[str]) -> str:
    """Recommend strategy based on remote diagnostic failure modes."""
    if "gradient_flow_blocked" in failure_modes:
        return "component_first_probe"
    if "optimizer_update_blocked" in failure_modes:
        return "surface_freeze_unfreeze"
    if "objective_or_update_instability" in failure_modes:
        return "geolens_regularized_probe"
    return "geolens_curriculum_probe"


def _assess_severity(m: GradientInstabilityMetrics, modes: list[str]) -> str:
    if "extreme_gradient_spike" in modes and "all_updates_rollback" in modes:
        return "critical"
    if "excessive_gradient_norm" in modes and "loss_not_improved" in modes:
        return "high"
    if "unstable_training" in modes:
        return "medium"
    return "low"


def _infer_likely_causes(m: GradientInstabilityMetrics, modes: list[str]) -> list[str]:
    causes: list[str] = []
    if "excessive_gradient_norm" in modes:
        causes.append("parameterization_too_sensitive")
        causes.append("geometric_path_non_smooth")
    if "all_updates_rollback" in modes:
        causes.append("optimizer_step_too_large")
        causes.append("objective_mismatch")
    if "no_parameter_change" in modes:
        causes.append("gradient_flow_blocked")
    if "loss_not_improved" in modes:
        causes.append("psf_normalization_issue")
    if not causes:
        causes.append("insufficient_data")
    return sorted(set(causes))


def _recommend_recoveries(modes: list[str]) -> list[str]:
    recs: list[str] = []
    if "excessive_gradient_norm" in modes or "extreme_gradient_spike" in modes:
        recs.append("reduce_parameter_dimension")
        recs.append("use_component_parameterization")
        recs.append("add_psf_regularization")
        recs.append("use_gradient_normalization")
    if "all_updates_rollback" in modes:
        recs.append("redesign_objective")
        recs.append("reduce_learning_rate")
    if "loss_not_improved" in modes:
        recs.append("objective_redesign_with_psf_regularization")
    if not recs:
        recs.append("run_probe_only")
    recs.append("report_negative_result")
    return recs


def _claim_implications(modes: list[str]) -> list[str]:
    return [
        "Gradient instability prevents stable optical updates",
        "native_lens_simulation claim not supported",
        "GeoLens geometric path requires parameterization redesign",
    ]


def _design_hints(modes: list[str], causes: list[str]) -> list[str]:
    hints: list[str] = []
    if "reduce_parameter_dimension" in modes or "parameterization_too_sensitive" in causes:
        hints.append("param_reduction_sweep_diagnostic")
    if "use_component_parameterization" in modes:
        hints.append("component_parameterization_design")
    if "redesign_objective" in modes or "objective_mismatch" in causes:
        hints.append("objective_psf_regularized_design")
    if "run_probe_only" in modes:
        hints.append("backend_probe_design")
    hints.append("negative_result_report_design")
    return hints
