"""Mock-real backend alignment reports."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


SMOKE_CAVEAT = (
    "Current DeepLens backend validates real base PSF generation and adapter-level encoder proxy behavior; "
    "it does not validate native physical encoder optimization."
)


def baseline_root() -> Path:
    return Path(os.getenv("OPTIRESEARCH_BASELINE_ROOT", "./workspace/baselines"))


def report_root() -> Path:
    return Path(os.getenv("OPTIRESEARCH_REPORT_ROOT", "./workspace/reports"))


def load_backend_baseline(backend: str) -> dict[str, Any]:
    path = baseline_root() / backend / "baseline_comparison.json"
    if not path.exists():
        legacy = baseline_root() / "baseline_comparison.json"
        if legacy.exists():
            payload = json.loads(legacy.read_text(encoding="utf-8"))
            if payload.get("backend") == backend:
                return payload
        return {"backend": backend, "runs": [], "warning": f"missing baseline: {path}"}
    return json.loads(path.read_text(encoding="utf-8"))


def compare_backend_metrics(left_backend: str, right_backend: str) -> dict[str, Any]:
    left = load_backend_baseline(left_backend)
    right = load_backend_baseline(right_backend)
    left_runs = {item.get("encoder_type"): item for item in left.get("runs", [])}
    right_runs = {item.get("encoder_type"): item for item in right.get("runs", [])}
    rows: list[dict[str, Any]] = []
    for encoder in sorted(set(left_runs) | set(right_runs)):
        left_item = left_runs.get(encoder, {})
        right_item = right_runs.get(encoder, {})
        left_metrics = left_item.get("metrics", {})
        right_metrics = right_item.get("metrics", {})
        rows.append(_comparison_row(encoder, left_item, right_item, left_metrics, right_metrics))
    rank_agreement = _rank_agreement(
        {item.get("encoder_type"): item.get("joint_tradeoff_score") for item in left.get("runs", [])},
        {item.get("encoder_type"): item.get("joint_tradeoff_score") for item in right.get("runs", [])},
    )
    return {
        "left_backend": left_backend,
        "right_backend": right_backend,
        "left_warning": left.get("warning"),
        "right_warning": right.get("warning"),
        "rows": rows,
        "summary": {
            "encoder_count": len(rows),
            "smoke_level_right": any(row.get("right_backend_capability_level") in {"smoke", "minimal"} for row in rows),
            "right_encoder_behavior_realized": all(row.get("right_encoder_behavior_realized") is True for row in rows) if rows else False,
            "rank_agreement": rank_agreement,
        },
        "proxy_realization": {
            "right_realization_levels": sorted({str(row.get("right_realization_level")) for row in rows if row.get("right_realization_level")}),
            "native_physical_validation": all(row.get("right_realization_level") == "native" for row in rows) if rows else False,
        },
        "claims_allowed": [
            "DeepLens adapter can produce encoder-specific baseline artifacts.",
            "controlled chromatic EDOF improves joint depth-spectral tradeoff under adapter-proxy DeepLens setting.",
        ],
        "claims_not_allowed": [
            "controlled chromatic EDOF is physically validated as best under DeepLens.",
            "adapter-proxy metrics prove native EDOF-HSI optical performance.",
        ],
        "caveat": SMOKE_CAVEAT,
    }


def export_backend_alignment_report(left_backend: str, right_backend: str) -> dict[str, Path]:
    comparison = compare_backend_metrics(left_backend, right_backend)
    root = report_root()
    root.mkdir(parents=True, exist_ok=True)
    name = f"backend_alignment_{left_backend.replace('_deeplens', '')}_vs_{right_backend}.json"
    json_path = root / name
    md_path = root / name.replace(".json", ".md")
    json_path.write_text(json.dumps(comparison, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(comparison), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def _comparison_row(
    encoder: str,
    left_item: dict[str, Any],
    right_item: dict[str, Any],
    left_metrics: dict[str, Any],
    right_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "encoder_type": encoder,
        "left_run_id": left_item.get("run_id"),
        "right_run_id": right_item.get("run_id"),
        "left_depth_planes": left_metrics.get("depth_planes"),
        "right_depth_planes": right_metrics.get("depth_planes"),
        "left_wavelength_bands": left_metrics.get("wavelength_bands"),
        "right_wavelength_bands": right_metrics.get("wavelength_bands"),
        "left_psf_depth_similarity": left_metrics.get("psf_depth_similarity"),
        "right_psf_depth_similarity": right_metrics.get("psf_depth_similarity"),
        "left_spectral_separability": left_metrics.get("spectral_separability"),
        "right_spectral_separability": right_metrics.get("spectral_separability"),
        "left_mtf_mean": left_metrics.get("mock_mtf_mean", left_metrics.get("deeplens_mtf_mean")),
        "right_mtf_mean": right_metrics.get("mock_mtf_mean", right_metrics.get("deeplens_mtf_mean")),
        "left_energy_efficiency": left_metrics.get("mock_energy_efficiency", left_metrics.get("deeplens_energy_efficiency")),
        "right_energy_efficiency": right_metrics.get("mock_energy_efficiency", right_metrics.get("deeplens_energy_efficiency")),
        "left_joint_score": left_item.get("joint_tradeoff_score"),
        "right_joint_score": right_item.get("joint_tradeoff_score"),
        "left_backend_capability_level": left_metrics.get("backend_capability_level"),
        "right_backend_capability_level": right_metrics.get("backend_capability_level"),
        "left_encoder_behavior_realized": left_metrics.get("encoder_behavior_realized"),
        "right_encoder_behavior_realized": right_metrics.get("encoder_behavior_realized"),
        "left_realization_level": left_metrics.get("encoder_behavior_realization_level"),
        "right_realization_level": right_metrics.get("encoder_behavior_realization_level"),
        "left_physical_validation_level": left_metrics.get("physical_validation_level"),
        "right_physical_validation_level": right_metrics.get("physical_validation_level"),
        "right_proxy_transform_name": right_metrics.get("proxy_transform_name"),
    }


def _rank_agreement(left_scores: dict[str | None, Any], right_scores: dict[str | None, Any]) -> float:
    encoders = [encoder for encoder in left_scores if encoder in right_scores and encoder is not None]
    if len(encoders) < 2:
        return 1.0 if encoders else 0.0
    left_rank = _rank_map({encoder: float(left_scores[encoder]) for encoder in encoders if left_scores[encoder] is not None})
    right_rank = _rank_map({encoder: float(right_scores[encoder]) for encoder in encoders if right_scores[encoder] is not None})
    pairs = 0
    agree = 0
    for idx, first in enumerate(encoders):
        for second in encoders[idx + 1 :]:
            if first not in left_rank or second not in left_rank or first not in right_rank or second not in right_rank:
                continue
            left_order = left_rank[first] < left_rank[second]
            right_order = right_rank[first] < right_rank[second]
            pairs += 1
            if left_order == right_order:
                agree += 1
    return round(agree / pairs, 6) if pairs else 1.0


def _rank_map(scores: dict[str, float]) -> dict[str, int]:
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return {encoder: rank for rank, (encoder, _score) in enumerate(ordered, start=1)}


def _markdown(comparison: dict[str, Any]) -> str:
    lines = [
        "# Backend Alignment Report",
        "",
        "## Summary",
        "",
        f"Left backend: `{comparison['left_backend']}`",
        f"Right backend: `{comparison['right_backend']}`",
        "",
        SMOKE_CAVEAT,
        "",
        "## Backend Capability Comparison",
        "",
        "| Backend | Capability Level | Encoder Behavior Realized | Realization Level | Physical Validation |",
        "|---|---|---|---|---|",
    ]
    first = comparison["rows"][0] if comparison["rows"] else {}
    lines.append(
        f"| {comparison['left_backend']} | {first.get('left_backend_capability_level', '')} | {first.get('left_encoder_behavior_realized', '')} | {first.get('left_realization_level', '')} | {first.get('left_physical_validation_level', '')} |"
    )
    lines.append(
        f"| {comparison['right_backend']} | {first.get('right_backend_capability_level', '')} | {first.get('right_encoder_behavior_realized', '')} | {first.get('right_realization_level', '')} | {first.get('right_physical_validation_level', '')} |"
    )
    lines.extend(
        [
            "",
            "## Proxy Realization",
            "",
            "DeepLens Phase 7 uses real DeepLens base PSF generation followed by explicit adapter-level proxy transforms for encoder-specific baseline behavior.",
            "",
            "## Native vs Proxy Distinction",
            "",
            "Adapter-proxy evidence can support adapter-level baseline claims. It cannot support native physical optimization claims.",
            "",
            "## Rank Alignment",
            "",
            f"Simple pairwise rank agreement: `{comparison['summary']['rank_agreement']}`",
            "",
            "## Metric Comparison",
            "",
            "| Encoder | Left Depth | Right Depth | Left Spectral | Right Spectral | Left MTF | Right MTF | Left Energy | Right Energy | Right Realization | Proxy Transform | Left Joint | Right Joint |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|",
        ]
    )
    for row in comparison["rows"]:
        lines.append(
            "| {encoder} | {ld} | {rd} | {ls} | {rs} | {lm} | {rm} | {le} | {re} | {realization} | {proxy} | {lj} | {rj} |".format(
                encoder=row["encoder_type"],
                ld=row.get("left_psf_depth_similarity", ""),
                rd=row.get("right_psf_depth_similarity", ""),
                ls=row.get("left_spectral_separability", ""),
                rs=row.get("right_spectral_separability", ""),
                lm=row.get("left_mtf_mean", ""),
                rm=row.get("right_mtf_mean", ""),
                le=row.get("left_energy_efficiency", ""),
                re=row.get("right_energy_efficiency", ""),
                realization=row.get("right_realization_level", ""),
                proxy=row.get("right_proxy_transform_name", ""),
                lj=row.get("left_joint_score", ""),
                rj=row.get("right_joint_score", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Mismatch Analysis",
            "",
            "Mock baselines intentionally encode encoder-specific behavior. Phase 7 DeepLens baselines encode comparable behavior through explicit adapter proxies on top of real DeepLens base PSF generation.",
            "",
            "## Evidence Caveats",
            "",
            "- DeepLens proxy results validate adapter integration, artifact registration, memory compilation, and adapter-level encoder-specific evidence.",
            "- They do not yet validate native DeepLens physical encoder optimization.",
            "",
            "## Encoder-Specific Behavior Status",
            "",
            "| Encoder | Realized | Level | Physical Validation |",
            "|---|---|---|---|",
        ]
    )
    for row in comparison["rows"]:
        lines.append(
            f"| {row['encoder_type']} | {row.get('right_encoder_behavior_realized', '')} | {row.get('right_realization_level', '')} | {row.get('right_physical_validation_level', '')} |"
        )
    lines.extend(
        [
            "",
            "## Claims Allowed / Not Allowed",
            "",
            "| Allowed | Not Allowed |",
            "|---|---|",
        ]
    )
    max_rows = max(len(comparison["claims_allowed"]), len(comparison["claims_not_allowed"]))
    for idx in range(max_rows):
        allowed = comparison["claims_allowed"][idx] if idx < len(comparison["claims_allowed"]) else ""
        not_allowed = comparison["claims_not_allowed"][idx] if idx < len(comparison["claims_not_allowed"]) else ""
        lines.append(f"| {allowed} | {not_allowed} |")
    lines.extend(
        [
            "",
            "## Next Actions Before True EDOF-HSI Optimization",
            "",
            "1. Map each encoder family to a concrete DeepLens lens or hybrid optical design.",
            "2. Replace replicated monochrome PSF bands with wavelength-aware HSI simulation.",
            "3. Add optimization-backed metrics before promoting encoder comparison claims.",
            "",
        ]
    )
    for key in ("left_warning", "right_warning"):
        if comparison.get(key):
            lines.append(f"Warning: {comparison[key]}")
    return "\n".join(lines)
