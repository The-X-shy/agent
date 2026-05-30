"""Multi-seed, multi-config reproducibility benchmark for stabilized native
GeoLens HSI optimization.

Wraps run_stabilized_native_geolens_hsi_loop() across a config matrix of
seeds x step_grid x spectral_angle_weights x grad_clip_norms, aggregating
per-config results into a statistical summary.
"""

from __future__ import annotations

import csv
import itertools
import json
import time
from pathlib import Path
from typing import Any

from optiresearch.analysis.native_geolens_benchmark_analysis import aggregate_config_results
from optiresearch.schemas.native_geolens_benchmark import (
    NativeGeoLensBenchmarkConfigResult,
    NativeGeoLensBenchmarkSpec,
    NativeGeoLensBenchmarkSummary,
    make_benchmark_id,
)
from optiresearch.schemas.native_geolens_stability import NativeGeoLensStabilitySpec
from optiresearch.schemas.stable_native_lens_hsi import make_stable_lens_id


def run_native_geolens_stability_benchmark(
    spec: NativeGeoLensBenchmarkSpec,
) -> NativeGeoLensBenchmarkSummary:
    """Run the reproducibility benchmark across all config combinations.

    Each config runs independently; a single config failure does not abort
    the entire benchmark.
    """
    from optiresearch.runtime.stable_native_lens_hsi_loop import (
        run_stabilized_native_geolens_hsi_loop,
    )

    benchmark_id = spec.benchmark_id or make_benchmark_id()
    config_results: list[NativeGeoLensBenchmarkConfigResult] = []

    # Build config matrix
    configs = list(
        itertools.product(
            spec.seeds,
            spec.step_grid,
            spec.spectral_angle_weights,
            spec.grad_clip_norms,
        )
    )
    if spec.max_configs and len(configs) > spec.max_configs:
        configs = configs[: spec.max_configs]

    for i, (seed, steps, sa_weight, grad_clip) in enumerate(configs):
        config_id = f"cfg_{i:03d}_s{seed}_t{steps}_w{sa_weight}_c{int(grad_clip)}"
        result = _run_single_config(
            config_id, seed, steps, sa_weight, grad_clip, spec,
            run_stabilized_native_geolens_hsi_loop,
        )
        config_results.append(result)

    # Aggregate
    summary_dict = aggregate_config_results(config_results, benchmark_id)

    summary = NativeGeoLensBenchmarkSummary(
        benchmark_id=benchmark_id,
        config_count=summary_dict["config_count"],
        completed_count=summary_dict["completed_count"],
        failed_count=summary_dict["failed_count"],
        seed_count=summary_dict["seed_count"],
        all_metrics_improved_count=summary_dict["all_metrics_improved_count"],
        all_metrics_improved_rate=summary_dict["all_metrics_improved_rate"],
        mse_improved_rate=summary_dict["mse_improved_rate"],
        psnr_improved_rate=summary_dict["psnr_improved_rate"],
        sam_improved_rate=summary_dict["sam_improved_rate"],
        mean_mse_delta=summary_dict.get("mean_mse_delta"),
        std_mse_delta=summary_dict.get("std_mse_delta"),
        mean_psnr_delta=summary_dict.get("mean_psnr_delta"),
        std_psnr_delta=summary_dict.get("std_psnr_delta"),
        mean_sam_delta=summary_dict.get("mean_sam_delta"),
        std_sam_delta=summary_dict.get("std_sam_delta"),
        mean_grad_norm_max=summary_dict.get("mean_grad_norm_max"),
        rollback_rate=summary_dict["rollback_rate"],
        best_config_id=summary_dict["best_config_id"],
        robust_config_family=summary_dict["robust_config_family"],
        claim_recommendation=summary_dict["claim_recommendation"],
        safe_wording=summary_dict["safe_wording"],
        blocked_claims=summary_dict["blocked_claims"],
        config_results=config_results,
    )

    if spec.save_artifacts:
        out_dir = Path("workspace/native_geolens_benchmarks") / benchmark_id
        out_dir.mkdir(parents=True, exist_ok=True)
        _save_artifacts(out_dir, spec, summary, config_results, benchmark_id)

    return summary


def _run_single_config(
    config_id: str,
    seed: int,
    steps: int,
    sa_weight: float,
    grad_clip: float,
    bench_spec: NativeGeoLensBenchmarkSpec,
    loop_fn: Any,
) -> NativeGeoLensBenchmarkConfigResult:
    """Run a single config, catching all errors."""
    errors: list[str] = []
    try:
        inner_spec = NativeGeoLensStabilitySpec(
            run_id=make_stable_lens_id("GeoLensCooke", "differentiable_linear"),
            candidate="GeoLensCooke",
            reconstructor="differentiable_linear",
            dataset=bench_spec.dataset,
            max_steps=steps,
            optical_warmup_steps=min(3, max(1, steps // 3)),
            spectral_angle_weight=sa_weight,
            optical_grad_clip=grad_clip,
            device=bench_spec.device,
            seed=seed,
            save_artifacts=False,
        )
        result = loop_fn(inner_spec)
    except Exception as exc:
        return NativeGeoLensBenchmarkConfigResult(
            config_id=config_id,
            seed=seed,
            steps=steps,
            spectral_angle_weight=sa_weight,
            grad_clip_norm=grad_clip,
            status="failed",
            errors=[str(exc)],
        )

    mse_before = result.mse_before
    mse_after = result.mse_after
    psnr_before = result.psnr_before
    psnr_after = result.psnr_after
    sam_before = result.sam_before
    sam_after = result.sam_after

    mse_delta = (mse_after - mse_before) if (mse_before is not None and mse_after is not None) else None
    psnr_delta = (psnr_after - psnr_before) if (psnr_before is not None and psnr_after is not None) else None
    sam_delta = (sam_after - sam_before) if (sam_before is not None and sam_after is not None) else None

    return NativeGeoLensBenchmarkConfigResult(
        config_id=config_id,
        seed=seed,
        steps=steps,
        spectral_angle_weight=sa_weight,
        grad_clip_norm=grad_clip,
        status=result.status,
        evidence_level=result.evidence_level,
        parameter_count=result.parameter_count,
        trainable_param_count=result.trainable_param_count,
        graph_connected=result.graph_connected,
        psf_requires_grad=result.psf_requires_grad,
        loss_requires_grad=result.loss_requires_grad,
        parameter_changed=result.optical_parameters_changed or False,
        accepted_update_count=result.accepted_update_count,
        rollback_count=result.rollback_count,
        rollback_reasons=result.rollback_reasons,
        mse_before=mse_before,
        mse_after=mse_after,
        mse_delta=mse_delta,
        mse_improved=(mse_delta is not None and mse_delta <= 0),
        psnr_before=psnr_before,
        psnr_after=psnr_after,
        psnr_delta=psnr_delta,
        psnr_improved=(psnr_delta is not None and psnr_delta >= 0),
        sam_before=sam_before,
        sam_after=sam_after,
        sam_delta=sam_delta,
        sam_improved=(sam_delta is not None and sam_delta <= 0),
        grad_norm_max=result.grad_norm_max,
        grad_norm_mean=result.grad_norm_mean,
        psf_centroid_shift=result.psf_centroid_shift,
        psf_width_shift=result.psf_width_shift,
        stability_score=result.stability_score,
        metric_tradeoff_summary=result.metric_tradeoff_summary,
        warnings=result.warnings,
        errors=errors,
    )


def _save_artifacts(
    out_dir: Path,
    spec: NativeGeoLensBenchmarkSpec,
    summary: NativeGeoLensBenchmarkSummary,
    config_results: list[NativeGeoLensBenchmarkConfigResult],
    benchmark_id: str,
) -> None:
    summary_data = summary.model_dump(mode="json")
    (out_dir / "benchmark_summary.json").write_text(
        json.dumps(summary_data, indent=2, default=str), encoding="utf-8",
    )
    (out_dir / "benchmark_spec.json").write_text(
        json.dumps(spec.model_dump(mode="json"), indent=2, default=str), encoding="utf-8",
    )

    # CSV
    csv_path = out_dir / "benchmark_results.csv"
    with open(csv_path, "w", newline="") as f:
        completed = [r for r in config_results if r.status == "succeeded"]
        if completed:
            writer = csv.DictWriter(f, fieldnames=list(completed[0].model_dump(mode="json").keys()))
            writer.writeheader()
            for r in config_results:
                writer.writerow(r.model_dump(mode="json"))

    # Trace JSON
    trace = {
        "benchmark_id": benchmark_id,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "configs_total": len(config_results),
        "configs_completed": summary.completed_count,
        "configs_failed": summary.failed_count,
    }
    (out_dir / "benchmark_trace.json").write_text(
        json.dumps(trace, indent=2), encoding="utf-8",
    )

    # Markdown report
    report_md = _benchmark_report_md(summary_data)
    (out_dir / "native_geolens_benchmark_report.md").write_text(report_md, encoding="utf-8")


def _benchmark_report_md(summary: dict[str, Any]) -> str:
    lines = [
        "# Native GeoLens Stability Benchmark Report",
        "",
        f"**Benchmark ID:** `{summary.get('benchmark_id', '')}`",
        "",
        "## 1. Summary",
        "",
        f"- Configs: {summary['config_count']} ({summary['completed_count']} completed, {summary['failed_count']} failed)",
        f"- Seeds: {summary['seed_count']}",
        "",
        "## 2. Improvement Rates",
        "",
        f"| Metric | Rate |",
        f"|--------|------|",
        f"| All metrics improved | {summary['all_metrics_improved_rate']:.1%} |",
        f"| MSE improved | {summary['mse_improved_rate']:.1%} |",
        f"| PSNR improved | {summary['psnr_improved_rate']:.1%} |",
        f"| SAM improved | {summary['sam_improved_rate']:.1%} |",
        "",
        "## 3. Metric Statistics",
        "",
        f"| Metric | Mean Δ | Std Δ |",
        f"|--------|--------|-------|",
        f"| MSE | {_fmt(summary.get('mean_mse_delta'))} | {_fmt(summary.get('std_mse_delta'))} |",
        f"| PSNR | {_fmt(summary.get('mean_psnr_delta'))} | {_fmt(summary.get('std_psnr_delta'))} |",
        f"| SAM | {_fmt(summary.get('mean_sam_delta'))} | {_fmt(summary.get('std_sam_delta'))} |",
        f"| Grad Norm Max | {_fmt(summary.get('mean_grad_norm_max'))} | — |",
        "",
        f"- Rollback rate: {summary.get('rollback_rate', 0):.1%}",
        "",
        "## 4. Best Config",
        f"- `{summary.get('best_config_id', '')}`",
        "",
        "## 5. Robust Config Family",
        f"- {summary.get('robust_config_family', '')}",
        "",
        "## 6. Claim Recommendation",
        f"- **Recommendation:** {summary.get('claim_recommendation', '')}",
        f"- **Safe wording:** {summary.get('safe_wording', '')}",
        "",
        "## 7. Blocked Claims",
    ]
    for bc in summary.get("blocked_claims", []):
        lines.append(f"- {bc}")
    return "\n".join(lines)


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)
