"""Baseline batch runner for mock optical encoders."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from optiresearch.runtime.graph import run_mvp_flow
from optiresearch.schemas.experiment import build_default_mock_edof_hsi_experiment
from optiresearch.memory.design_rule import DesignRuleManager
from optiresearch.storage.sqlite_store import SQLiteStore


ENCODER_TYPES = [
    "conventional",
    "achromatic",
    "edof",
    "chromatic_coded",
    "controlled_chromatic_edof",
]


def run_baseline_batch(
    objective: str,
    workspace_id: str = "default",
    output_root: Optional[Path] = None,
    backend: str = "mock_deeplens",
    encoder: str = "all",
    realization: str = "auto",
) -> dict[str, Any]:
    """Run one MVP flow per encoder and write comparison reports."""

    if backend not in {"mock_deeplens", "deeplens"}:
        raise ValueError(f"Unsupported backend: {backend}")
    if encoder != "all" and encoder not in ENCODER_TYPES:
        raise ValueError(f"Unsupported encoder: {encoder}")
    output_root = output_root or (Path(os.getenv("OPTIRESEARCH_BASELINE_ROOT", "./workspace/baselines")) / backend)
    runs: list[dict[str, Any]] = []
    encoder_types = ENCODER_TYPES if encoder == "all" else [encoder]
    for encoder_type in encoder_types:
        experiment = build_default_mock_edof_hsi_experiment(objective, encoder_type=encoder_type)
        if backend == "deeplens":
            experiment = experiment.model_copy(
                update={
                    "backend": "deeplens",
                    "metadata": {**experiment.metadata, "backend": "deeplens"},
                },
                deep=True,
            )
        result = run_mvp_flow(
            f"{objective} [{encoder_type}]",
            workspace_id=workspace_id,
            experiment_spec=experiment,
            backend=backend,
            realization=realization,
        )
        metrics = result["run_memory"]["best_metrics"]
        runs.append(
            {
                "encoder_type": encoder_type,
                "run_id": result["run_id"],
                "artifact_ids": result["artifact_ids"],
                "claim_ids": [claim["claim_id"] for claim in result["claims"]],
                "metrics": metrics,
                "claim_statuses": {claim["text"]: claim["status"] for claim in result["claims"]},
                "errors": result.get("errors", []),
                "joint_tradeoff_score": _joint_tradeoff(metrics),
            }
        )
    best = max(runs, key=lambda item: item["joint_tradeoff_score"])
    report = {
        "objective": objective,
        "workspace_id": workspace_id,
        "backend": backend,
        "realization": realization,
        "runs": runs,
        "best_joint_tradeoff": best,
    }
    rules = DesignRuleManager(SQLiteStore()).compile_from_claims() if backend == "mock_deeplens" else []
    report["design_rule_ids"] = [rule.rule_id for rule in rules]
    _write_reports(report, output_root)
    return report


def _joint_tradeoff(metrics: dict[str, Any]) -> float:
    depth = float(metrics.get("psf_depth_similarity", 0.0))
    spectral = float(metrics.get("spectral_separability", 0.0))
    mtf = float(metrics.get("mock_mtf_mean", metrics.get("deeplens_mtf_mean", 0.0)))
    energy = float(metrics.get("mock_energy_efficiency", metrics.get("deeplens_energy_efficiency", 0.0)))
    return round(0.35 * depth + 0.35 * spectral + 0.15 * mtf + 0.15 * energy, 6)


def _write_reports(report: dict[str, Any], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "baseline_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Baseline Comparison",
        "",
        f"Objective: {report['objective']}",
        f"Backend: {report['backend']}",
        "",
        "| Encoder | Run ID | Depth Similarity | Spectral Separability | MTF | Energy | Capability | Encoder Realized | Realization Level | Selected Realization | Semi-native Attempted | Semi-native Succeeded | Proxy Fallback | Claim Scope | Physical Validation | Proxy Transform | Caveat | Joint |",
        "|---|---|---:|---:|---:|---:|---|---|---|---|---|---|---|---|---|---|---|---:|",
    ]
    if report["backend"] == "deeplens":
        lines.extend(
            [
                "",
                "Current DeepLens baseline uses real DeepLens base PSF generation plus adapter-level encoder proxy transforms. It is not native physical encoder optimization.",
                "",
            ]
        )
    for item in report["runs"]:
        metrics = item["metrics"]
        lines.append(
            "| {encoder} | {run_id} | {depth} | {spectral} | {mtf} | {energy} | {capability} | {realized} | {realization_level} | {selected} | {attempted} | {succeeded} | {fallback} | {claim_scope} | {physical} | {proxy} | {caveat} | {joint} |".format(
                encoder=item["encoder_type"],
                run_id=item["run_id"],
                depth=metrics.get("psf_depth_similarity"),
                spectral=metrics.get("spectral_separability"),
                mtf=metrics.get("mock_mtf_mean", metrics.get("deeplens_mtf_mean")),
                energy=metrics.get("mock_energy_efficiency", metrics.get("deeplens_energy_efficiency")),
                capability=metrics.get("backend_capability_level"),
                realized=metrics.get("encoder_behavior_realized"),
                realization_level=metrics.get("encoder_behavior_realization_level", ""),
                selected=metrics.get("selected_realization_level", ""),
                attempted=metrics.get("semi_native_attempted", ""),
                succeeded=metrics.get("semi_native_succeeded", ""),
                fallback=metrics.get("proxy_fallback_used", ""),
                claim_scope=metrics.get("claim_scope", ""),
                physical=metrics.get("physical_validation_level", ""),
                proxy=metrics.get("proxy_transform_name", ""),
                caveat=_baseline_caveat(metrics),
                joint=item["joint_tradeoff_score"],
            )
        )
    best = report["best_joint_tradeoff"]
    lines.extend(["", f"Best joint tradeoff: `{best['encoder_type']}` ({best['run_id']}).", ""])
    (output_root / "baseline_comparison.md").write_text("\n".join(lines), encoding="utf-8")


def _baseline_caveat(metrics: dict[str, Any]) -> str:
    if metrics.get("selected_realization_level") == "semi_native":
        return "semi-native evidence, not native optimization"
    if metrics.get("encoder_behavior_realization_level") == "adapter_proxy":
        return "adapter proxy, not native physical validation"
    if metrics.get("backend_capability_level") in {"smoke", "minimal"}:
        return "smoke-level integration evidence"
    if metrics.get("backend_capability_level") == "mock":
        return "mock evidence"
    return ""
