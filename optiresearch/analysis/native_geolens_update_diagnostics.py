"""Diagnose native GeoLens HSI update instability for Phase 35 stabilization.

Analyzes Phase 34 (or any) native GeoLens HSI run results and produces
diagnostic recommendations for stabilizing optical updates.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def diagnose_native_geolens_update(run_dir: str) -> dict[str, Any]:
    root = Path(run_dir)
    result_file = root / "result.json"
    if not result_file.exists():
        for alt in root.rglob("result.json"):
            result_file = alt
            break

    result = _load_json(result_file)
    if not result:
        return {"error": f"No result.json found in {run_dir}"}

    opt_gn_max = result.get("optical_gradient_norm_max")
    opt_gn_mean = result.get("optical_gradient_norm_mean")
    recon_gn_max = result.get("recon_gradient_norm_max")
    recon_gn_mean = result.get("recon_gradient_norm_mean")

    accepted = result.get("accepted_update_count", 0)
    rejected = result.get("rejected_update_count", 0)
    rollbacks = result.get("rollback_count", 0)
    opt_changed = result.get("optical_parameters_changed", False)
    stable = result.get("stable_training_succeeded", False)

    loss_b = result.get("reconstruction_loss_before")
    loss_a = result.get("reconstruction_loss_after")
    mse_b = result.get("mse_before")
    mse_a = result.get("mse_after")
    psnr_b = result.get("psnr_before")
    psnr_a = result.get("psnr_after")
    sam_b = result.get("sam_before")
    sam_a = result.get("sam_after")

    psf_energy_delta = result.get("psf_energy_delta", 0)
    psf_width_delta = result.get("psf_width_delta", 0)

    grad_risk = "high" if opt_gn_max and opt_gn_max > 100 else (
        "medium" if opt_gn_max and opt_gn_max > 10 else "low"
    )

    update_strength = opt_gn_mean * (1e-6) if opt_gn_mean else None
    overshoot_risk = "high" if update_strength and update_strength > 1e-2 else (
        "medium" if update_strength and update_strength > 1e-3 else "low"
    )

    loss_changed = loss_a is not None and loss_b is not None
    loss_decreased = loss_changed and loss_a <= loss_b
    loss_delta_pct = (
        100 * (loss_a - loss_b) / (loss_b + 1e-8) if loss_changed else None
    )

    diagnosis: dict[str, Any] = {
        "run_id": result.get("run_id"),
        "status": result.get("status"),
        "evidence_level": result.get("evidence_level"),
        "gradient_analysis": {
            "optical_gradient_norm_max": opt_gn_max,
            "optical_gradient_norm_mean": opt_gn_mean,
            "recon_gradient_norm_max": recon_gn_max,
            "recon_gradient_norm_mean": recon_gn_mean,
            "gradient_ratio": (opt_gn_mean / recon_gn_mean) if opt_gn_mean and recon_gn_mean else None,
            "optical_gradient_risk": grad_risk,
        },
        "update_analysis": {
            "estimated_update_magnitude": update_strength,
            "overshoot_risk": overshoot_risk,
        },
        "loss_analysis": {
            "reconstruction_loss_before": loss_b,
            "reconstruction_loss_after": loss_a,
            "loss_delta_pct": loss_delta_pct,
            "loss_decreased": loss_decreased,
            "mse_before": mse_b,
            "mse_after": mse_a,
            "psnr_before": psnr_b,
            "psnr_after": psnr_a,
            "sam_before": sam_b,
            "sam_after": sam_a,
        },
        "rollback_analysis": {
            "accepted_update_count": accepted,
            "rejected_update_count": rejected,
            "rollback_count": rollbacks,
            "optical_parameters_changed": opt_changed,
            "stable_training_succeeded": stable,
            "all_updates_rejected": accepted == 0 and rejected > 0,
            "no_updates_attempted": accepted == 0 and rejected == 0,
        },
        "psf_stability": {
            "psf_energy_delta": psf_energy_delta,
            "psf_width_delta": psf_width_delta,
        },
    }

    main_causes: list[str] = []
    if grad_risk == "high":
        main_causes.append(
            f"Optical gradient norm too large (max={opt_gn_max:.1f}). "
            "Even with lr=1e-6, the update overshoots. "
            "Need stronger gradient clipping (0.1 or 0.01) or trust-region scaling."
        )
    if accepted == 0 and rejected > 0:
        main_causes.append(
            f"All {rejected} optical updates rejected by rollback. "
            "Loss increased after every update attempt. "
            "Need smaller effective step size or accept_tolerance > 0."
        )
    if loss_decreased and accepted == 0:
        main_causes.append(
            "Final loss decreased despite all updates being rejected — "
            "improvement is from reconstructor-only steps. Optical updates not contributing."
        )
    if psf_energy_delta and abs(psf_energy_delta) > 0.1:
        main_causes.append(
            f"PSF energy changed by {psf_energy_delta:.4f}. "
            "Optical updates may be destabilizing the PSF."
        )
    diagnosis["main_causes"] = main_causes

    recommendations: list[dict[str, Any]] = []
    if grad_risk == "high":
        recommendations.append({
            "action": "reduce_optical_grad_clip",
            "current": 1.0,
            "recommended_range": [0.01, 0.1],
            "rationale": f"Clip optical gradients to 0.01-0.1 (current norm: {opt_gn_max:.1f})",
        })
        recommendations.append({
            "action": "reduce_optical_lr",
            "current": 1e-6,
            "recommended_range": [1e-7, 5e-8, 1e-8],
            "rationale": "Lower lr compensates for high gradient sensitivity",
        })
    if accepted == 0:
        recommendations.append({
            "action": "enable_trust_region",
            "recommended": True,
            "rationale": "Scale down parameter updates that exceed max_optical_param_delta",
        })
        recommendations.append({
            "action": "add_accept_tolerance",
            "recommended": 1e-6,
            "rationale": "Allow sub-epsilon loss increases as exploratory updates",
        })
    recommendations.append({
        "action": "enable_psf_stability_gating",
        "recommended": True,
        "rationale": "Reject updates that cause large PSF energy or centroid shifts",
    })
    diagnosis["recommendations"] = recommendations

    suggested_configs = _generate_suggested_configs(opt_gn_max or 4098, accepted, rejected)
    diagnosis["suggested_sweep_configs"] = suggested_configs

    report_dir = Path("workspace/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "native_geolens_update_diagnostics.json").write_text(
        json.dumps(diagnosis, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    (report_dir / "native_geolens_update_diagnostics.md").write_text(
        _diagnosis_md(diagnosis), encoding="utf-8"
    )

    return diagnosis


def _generate_suggested_configs(
    opt_gn_max: float, accepted: int, rejected: int
) -> list[dict[str, Any]]:
    configs = []
    lr_values = [1e-6, 5e-7, 1e-7, 5e-8, 1e-8]
    clip_values = [1.0, 0.1, 0.01]
    for lr in lr_values:
        for clip in clip_values:
            effective_update = lr * min(opt_gn_max, clip)
            risk = "high" if effective_update > 1e-2 else (
                "medium" if effective_update > 1e-3 else "low"
            )
            configs.append({
                "optical_lr": lr,
                "optical_grad_clip": clip,
                "effective_max_update": effective_update,
                "overshoot_risk": risk,
            })
    configs.sort(key=lambda c: c["effective_max_update"])
    return configs


def _diagnosis_md(d: dict[str, Any]) -> str:
    ga = d.get("gradient_analysis", {})
    ua = d.get("update_analysis", {})
    la = d.get("loss_analysis", {})
    ra = d.get("rollback_analysis", {})
    ps = d.get("psf_stability", {})

    lines = [
        "# Native GeoLens Update Diagnostics",
        "",
        f"**Run:** {d.get('run_id')} — Status: {d.get('status')}",
        f"**Evidence Level:** {d.get('evidence_level')}",
        "",
        "## Gradient Analysis",
        f"- optical_gradient_norm_max: {ga.get('optical_gradient_norm_max')}",
        f"- optical_gradient_norm_mean: {ga.get('optical_gradient_norm_mean')}",
        f"- recon_gradient_norm_max: {ga.get('recon_gradient_norm_max')}",
        f"- recon_gradient_norm_mean: {ga.get('recon_gradient_norm_mean')}",
        f"- gradient_ratio (opt/recon): {ga.get('gradient_ratio')}",
        f"- optical_gradient_risk: **{ga.get('optical_gradient_risk')}**",
        "",
        "## Update Analysis",
        f"- estimated_update_magnitude: {ua.get('estimated_update_magnitude')}",
        f"- overshoot_risk: **{ua.get('overshoot_risk')}**",
        "",
        "## Loss Analysis",
        f"- reconstruction_loss: {la.get('reconstruction_loss_before')} → {la.get('reconstruction_loss_after')}",
        f"- loss_delta_pct: {la.get('loss_delta_pct')}",
        f"- loss_decreased: {la.get('loss_decreased')}",
        f"- mse: {la.get('mse_before')} → {la.get('mse_after')}",
        f"- psnr: {la.get('psnr_before')} → {la.get('psnr_after')}",
        f"- sam: {la.get('sam_before')} → {la.get('sam_after')}",
        "",
        "## Rollback Analysis",
        f"- accepted: {ra.get('accepted_update_count')}",
        f"- rejected: {ra.get('rejected_update_count')}",
        f"- rollbacks: {ra.get('rollback_count')}",
        f"- optical_parameters_changed: {ra.get('optical_parameters_changed')}",
        f"- stable_training_succeeded: {ra.get('stable_training_succeeded')}",
        f"- all_updates_rejected: {ra.get('all_updates_rejected')}",
        "",
        "## PSF Stability",
        f"- psf_energy_delta: {ps.get('psf_energy_delta')}",
        f"- psf_width_delta: {ps.get('psf_width_delta')}",
        "",
        "## Main Causes",
    ]
    for c in d.get("main_causes", []):
        lines.append(f"- {c}")

    lines.extend(["", "## Recommendations"])
    for r in d.get("recommendations", []):
        lines.append(f"- **{r['action']}**: {r['rationale']}")

    lines.extend(["", "## Suggested Sweep Configs"])
    lines.append("| optical_lr | optical_grad_clip | effective_max_update | overshoot_risk |")
    lines.append("|---|---|---|---|")
    for c in d.get("suggested_sweep_configs", [])[:15]:
        lines.append(
            f"| {c['optical_lr']} | {c['optical_grad_clip']} | "
            f"{c['effective_max_update']:.2e} | {c['overshoot_risk']} |"
        )

    return "\n".join(lines) + "\n"


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
