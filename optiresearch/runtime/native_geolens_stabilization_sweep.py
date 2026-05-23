"""Native GeoLens stabilization sweep for Phase 35.

Systematically tests optical_lr, optical_grad_clip, and trust_region
combinations to find configurations that produce accepted optical updates.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from optiresearch.runtime.stable_native_lens_hsi_loop import (
    run_stable_native_lens_hsi_codesign,
)
from optiresearch.schemas.stable_native_lens_hsi import (
    StableNativeLensHSISpec,
    make_stable_lens_id,
)


def run_native_geolens_stabilization_sweep(
    lens_file: str = "auto:cooke",
    dataset: str = "synthetic",
    reconstructor: str = "differentiable_linear",
    device: str = "cpu",
    save_artifacts: bool = True,
) -> dict[str, Any]:
    sweep_id = f"geolens_stabilization_{int(time.time())}"
    base = make_stable_lens_id("GeoLensCooke", reconstructor)

    configs = _build_config_matrix()
    results: dict[str, dict[str, Any]] = {}

    for name, overrides in configs.items():
        spec = StableNativeLensHSISpec(
            run_id=f"{base}_{name}",
            candidate="GeoLensCooke",
            reconstructor=reconstructor,
            dataset=dataset,
            max_steps=5,
            optical_lr=overrides.get("optical_lr", 1e-6),
            optical_grad_clip=overrides.get("optical_grad_clip", 1.0),
            rollback_on_loss_increase=True,
            trust_region_enabled=overrides.get("trust_region_enabled", False),
            max_optical_param_delta=overrides.get("max_optical_param_delta", 1e-3),
            rollback_on_psf_instability=overrides.get("rollback_on_psf_instability", False),
            max_psf_energy_delta=overrides.get("max_psf_energy_delta", 0.1),
            max_psf_width_delta=overrides.get("max_psf_width_delta", 2.0),
            accept_tolerance=overrides.get("accept_tolerance", 0.0),
            device=device,
            full_wave_optics=False,
            phase_to_fft_proxy_used=False,
            save_artifacts=False,
        )
        r = run_stable_native_lens_hsi_codesign(spec)
        results[name] = {
            "status": r.status,
            "reconstruction_loss_before": r.reconstruction_loss_before,
            "reconstruction_loss_after": r.reconstruction_loss_after,
            "best_reconstruction_loss": r.best_reconstruction_loss,
            "optical_gradient_norm_max": r.optical_gradient_norm_max,
            "optical_gradient_norm_mean": r.optical_gradient_norm_mean,
            "accepted_update_count": r.accepted_update_count,
            "rejected_update_count": r.rejected_update_count,
            "rollback_count": r.rollback_count,
            "optical_parameters_changed": r.optical_parameters_changed,
            "optical_parameter_delta_max": r.optical_parameter_delta_max,
            "psf_energy_delta": r.psf_energy_delta,
            "psf_width_delta": r.psf_width_delta,
            "stable_training_succeeded": r.stable_training_succeeded,
            "trust_region_activated": r.trust_region_activated,
            "evidence_level": r.evidence_level,
            "error_code": r.error_code,
            "caveats": r.caveats,
            "overrides": overrides,
        }

    best_config_name = _find_best(results)
    best_config = configs.get(best_config_name, {}) if best_config_name else {}
    best_result = results.get(best_config_name, {})

    summary: dict[str, Any] = {
        "sweep_id": sweep_id,
        "run_id_base": base,
        "lens_file": lens_file,
        "dataset": dataset,
        "reconstructor": reconstructor,
        "device": device,
        "configs_tested": len(results),
        "configs_succeeded": sum(1 for r in results.values() if r["status"] == "succeeded"),
        "configs_with_accepted_updates": sum(1 for r in results.values() if r["accepted_update_count"] > 0),
        "best_config_name": best_config_name,
        "best_config": best_config,
        "best_result": best_result,
        "results": {k: _serialize_result(v) for k, v in results.items()},
    }

    if save_artifacts:
        out_dir = Path("workspace/native_geolens_stabilization") / sweep_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "sweep_spec.json").write_text(
            json.dumps({"sweep_id": sweep_id, "base": base, "configs": _serialize_configs(configs)},
                       indent=2, ensure_ascii=False), encoding="utf-8")
        (out_dir / "sweep_results.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        (out_dir / "best_config.json").write_text(
            json.dumps({"name": best_config_name, "config": best_config, "result": best_result},
                       indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        (out_dir / "sweep_table.md").write_text(_sweep_table_md(sweep_id, results, best_config_name), encoding="utf-8")
        (out_dir / "report.md").write_text(_sweep_report_md(summary), encoding="utf-8")
        summary["output_dir"] = str(out_dir)

    return summary


def _build_config_matrix() -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}
    lr_values = [1e-6, 5e-7, 1e-7, 5e-8, 1e-8]
    clip_values = [1.0, 0.1, 0.01]
    trust_configs = [
        {"trust_region_enabled": True, "max_optical_param_delta": 1e-3,
         "rollback_on_psf_instability": True, "max_psf_energy_delta": 0.1,
         "max_psf_width_delta": 2.0, "accept_tolerance": 1e-6},
        {"trust_region_enabled": True, "max_optical_param_delta": 1e-4,
         "rollback_on_psf_instability": True, "max_psf_energy_delta": 0.1,
         "max_psf_width_delta": 2.0, "accept_tolerance": 1e-6},
    ]
    for lr in lr_values:
        for clip in clip_values:
            for tc in trust_configs:
                name = f"lr{lr}_clip{clip}_tr{tc['max_optical_param_delta']}"
                configs[name] = {
                    "optical_lr": lr,
                    "optical_grad_clip": clip,
                    **tc,
                }
    return configs


def _find_best(results: dict[str, dict[str, Any]]) -> str | None:
    # Priority 1: accepted_update_count > 0 AND stable_training_succeeded
    # Priority 2: accepted_update_count > 0
    # Priority 3: min reconstruction_loss_after
    # Priority 4: min optical_gradient_norm
    candidates = [(k, v) for k, v in results.items() if v["status"] == "succeeded"]
    if not candidates:
        return None

    def _score(name: str, r: dict[str, Any]) -> tuple[int, int, float, float]:
        accepted = r.get("accepted_update_count", 0)
        stable = 1 if r.get("stable_training_succeeded") else 0
        loss = r.get("reconstruction_loss_after") or float("inf")
        gn = r.get("optical_gradient_norm_max") or float("inf")
        # Negative score: higher is better
        accepted_score = 1000 if accepted > 0 else 0
        stable_score = 100 if stable else 0
        return (accepted_score + stable_score, accepted, -loss, -gn)

    best = max(candidates, key=lambda kv: _score(kv[0], kv[1]))
    return best[0]


def _serialize_result(r: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in r.items() if k != "overrides"}


def _serialize_configs(configs: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {k: {sk: sv for sk, sv in v.items()} for k, v in configs.items()}


def _sweep_table_md(sweep_id: str, results: dict[str, dict[str, Any]], best_name: str | None) -> str:
    lines = [
        f"# Stabilization Sweep: {sweep_id}",
        "",
        "| Config | Status | Loss Before | Loss After | Accept | Reject | Rollback | Opt GN Max | Stable | Best |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in sorted(results.keys()):
        r = results[name]
        marker = "⭐" if name == best_name else ""
        lines.append(
            f"| {name} | {r['status']} | {_fmt(r.get('reconstruction_loss_before'))} | "
            f"{_fmt(r.get('reconstruction_loss_after'))} | {r.get('accepted_update_count', 0)} | "
            f"{r.get('rejected_update_count', 0)} | {r.get('rollback_count', 0)} | "
            f"{_fmt(r.get('optical_gradient_norm_max'))} | {r.get('stable_training_succeeded')} | {marker} |"
        )
    return "\n".join(lines) + "\n"


def _sweep_report_md(summary: dict[str, Any]) -> str:
    lines = [
        "# Native GeoLens Stabilization Sweep Report",
        "",
        f"**Sweep ID:** `{summary.get('sweep_id')}`",
        f"**Configs Tested:** {summary.get('configs_tested')}",
        f"**Configs Succeeded:** {summary.get('configs_succeeded')}",
        f"**Configs with Accepted Updates:** {summary.get('configs_with_accepted_updates')}",
        "",
        "## Best Config",
        f"**Name:** {summary.get('best_config_name')}",
    ]
    best_config = summary.get("best_config", {})
    if best_config:
        for k, v in best_config.items():
            lines.append(f"- {k}: {v}")
    best_result = summary.get("best_result", {})
    if best_result:
        lines.extend(["", "### Best Result", ""])
        for k, v in best_result.items():
            if k not in ("overrides", "caveats"):
                lines.append(f"- {k}: {v}")
    return "\n".join(lines) + "\n"


def _fmt(val: Any) -> str:
    if val is None:
        return "-"
    if isinstance(val, float):
        return f"{val:.6f}"
    return str(val)
