"""HSI dataset/reconstructor matrix runner and matrix-level evidence review."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from optiresearch.hsi.reconstruction import torch_available
from optiresearch.memory.claim_evidence import ClaimEvidenceManager
from optiresearch.memory.compiler import MemoryCompiler
from optiresearch.memory.design_rule import DesignRuleManager, compile_rules_from_hsi_matrix
from optiresearch.memory.schemas import EvidenceEdge, MetaTrace, make_deterministic_id, make_trace_id
from optiresearch.memory.meta_trace import MetaTraceWriter
from optiresearch.runtime.baselines import ENCODER_TYPES
from optiresearch.runtime.hsi_pipeline import run_hsi_reconstruction_flow
from optiresearch.storage.file_artifact_store import FileArtifactStore
from optiresearch.storage.sqlite_store import SQLiteStore


def run_hsi_matrix(
    datasets: list[str] | None = None,
    backends: list[str] | None = None,
    encoders: list[str] | None = None,
    reconstructors: list[str] | None = None,
    forward_modes: list[str] | None = None,
    objective: str = "Compare encoder ranking across HSI reconstructors",
    workspace_id: str = "default",
    dataset_path: str | None = None,
    use_optical_feature_maps: bool = False,
    tiny_cnn_epochs: int = 5,
    tiny_cnn_hidden: int = 32,
    device: str = "cpu",
) -> dict[str, Any]:
    datasets = datasets or ["synthetic"]
    backends = backends or ["mock_deeplens"]
    encoders = encoders or ENCODER_TYPES
    reconstructors = reconstructors or ["optical_conditioned_linear", "tiny_cnn"]
    forward_modes = forward_modes or ["depth_spectral_coded"]

    matrix_id = make_deterministic_id("hsi_matrix", objective, datasets, backends, encoders, reconstructors, forward_modes)
    root = Path(os.getenv("OPTIRESEARCH_HSI_ROOT", "./workspace/hsi")) / "matrix" / matrix_id
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for dataset in datasets:
        for backend in backends:
            for reconstructor in reconstructors:
                for forward_mode in forward_modes:
                    if reconstructor in {"tiny_cnn", "unet_tiny"} and not torch_available():
                        for encoder in encoders:
                            rows.append(_skipped_row(dataset, backend, encoder, reconstructor, forward_mode, "TORCH_NOT_AVAILABLE"))
                        continue
                    for encoder in encoders:
                        result = run_hsi_reconstruction_flow(
                            f"{objective} [{dataset}/{backend}/{reconstructor}/{encoder}/{forward_mode}]",
                            backend=backend,
                            encoder_type=encoder,
                            workspace_id=workspace_id,
                            dataset=dataset,
                            dataset_path=dataset_path,
                            forward_mode=forward_mode,
                            reconstructor=reconstructor,
                            use_optical_feature_maps=use_optical_feature_maps,
                            tiny_cnn_epochs=tiny_cnn_epochs,
                            tiny_cnn_hidden=tiny_cnn_hidden,
                            device=device,
                        )
                        if result.get("status") == "error":
                            rows.append(_skipped_row(dataset, backend, encoder, reconstructor, forward_mode, result.get("error_code", "RUN_FAILED")))
                            continue
                        if result.get("status") == "skipped":
                            rows.append(_skipped_row(dataset, backend, encoder, reconstructor, forward_mode, result.get("error_code", "RECONSTRUCTOR_SKIPPED")))
                            continue
                        metrics = result["metrics"]
                        rows.append(
                            {
                                "status": "succeeded",
                                "dataset": dataset,
                                "backend": backend,
                                "encoder": encoder,
                                "reconstructor": reconstructor,
                                "forward_mode": forward_mode,
                                "run_id": result["run_id"],
                                "psnr": metrics.get("PSNR"),
                                "ssim": metrics.get("SSIM"),
                                "sam": metrics.get("SAM"),
                                "ergas": metrics.get("ERGAS"),
                                "worst_depth_sam": metrics.get("worst_depth_SAM"),
                                "reconstruction_score": reconstruction_score(metrics),
                                "rank_within_group": None,
                                "evidence_level": result.get("evidence_level"),
                                "caveat": _caveat(dataset, backend, reconstructor),
                            }
                        )

    _rank_rows(rows)
    summary = _matrix_summary(matrix_id, rows)
    result_payload = {"matrix_id": matrix_id, "objective": objective, "rows": rows, "summary": summary}
    results_path = root / "hsi_matrix_results.json"
    summary_path = root / "hsi_matrix_summary.json"
    md_path = root / "hsi_matrix_results.md"
    results_path.write_text(json.dumps(result_payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    md_path.write_text(_matrix_markdown(result_payload), encoding="utf-8")

    store = SQLiteStore()
    store.init_db()
    artifact_store = FileArtifactStore(store=store)
    trace = MetaTraceWriter(store).write_trace(_matrix_trace(workspace_id, matrix_id, objective, rows))
    registered = [
        artifact_store.register_file(
            path,
            workspace_id=workspace_id,
            run_id=matrix_id,
            trace_id=trace.trace_id,
            producer="HSIMatrixRunner",
            metadata={"filename": path.name, "artifact_type": "hsi_matrix", "evidence_domain": "hsi_matrix"},
            metrics={},
        )
        for path in (results_path, md_path, summary_path)
    ]
    result_payload["artifact_ids"] = [item.artifact_id for item in registered]
    result_payload["artifact_uris"] = [item.uri for item in registered]
    claims = evaluate_matrix_claims(result_payload, ClaimEvidenceManager(store, workspace_id=workspace_id))
    result_payload["claim_ids"] = [claim.claim_id for claim in claims]
    result_payload["claims"] = [claim.model_dump(mode="json") for claim in claims]
    rules = compile_rules_from_hsi_matrix(result_payload)
    rule_manager = DesignRuleManager(store, workspace_id=workspace_id)
    saved_rules = [rule_manager.save(rule) for rule in rules]
    result_payload["design_rule_ids"] = [rule.rule_id for rule in saved_rules]
    result_payload["design_rules"] = [rule.model_dump(mode="json") for rule in saved_rules]
    result_payload["run_memory"] = MemoryCompiler(store=store, artifact_store=artifact_store).compile_run_memory(matrix_id).model_dump(mode="json")
    results_path.write_text(json.dumps(result_payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    final_ref = artifact_store.register_file(
        results_path,
        workspace_id=workspace_id,
        run_id=matrix_id,
        trace_id=trace.trace_id,
        producer="HSIMatrixRunner",
        metadata={"filename": results_path.name, "artifact_type": "hsi_matrix", "evidence_domain": "hsi_matrix", "final": True},
        metrics={},
    )
    result_payload["artifact_ids"].append(final_ref.artifact_id)
    result_payload["artifact_uris"].append(final_ref.uri)
    return result_payload


def evaluate_matrix_claims(matrix_result: dict[str, Any], manager: ClaimEvidenceManager | None = None) -> list:
    manager = manager or ClaimEvidenceManager()
    rows = matrix_result.get("rows", [])
    artifact_id = (matrix_result.get("artifact_ids") or ["matrix_result"])[0]
    specs = [
        _claim_controlled_tiny(matrix_result, rows),
        _claim_achromatic_all(matrix_result, rows),
        _claim_chromatic_requires_stronger(matrix_result, rows),
    ]
    claims = []
    for text, status, score, metadata in specs:
        claim = manager.create_claim(
            text,
            scope={
                "backend": "mock_deeplens",
                "evidence_domain": "hsi_matrix",
                "matrix_id": matrix_result.get("matrix_id"),
                "claim_scope": metadata.get("dataset_scope"),
            },
        )
        claim.status = status
        claim.support_score = score
        claim.review_status = "reviewed"
        claim.metadata.update(metadata)
        claim.metadata.setdefault("evidence_level", "matrix_synthetic_mock" if "synthetic" in str(metadata) else "matrix")
        relation = "supports" if status == "supported" else "qualifies"
        claim.support_edges.append(
            EvidenceEdge(
                artifact_id=artifact_id,
                trace_id=None,
                metric_name="rank_within_group",
                metric_value=str(metadata.get("rank_comparison", {})),
                relation=relation,
                score=score,
                rationale=f"Matrix-level rank comparison for claim status {status}.",
            )
        )
        manager._save(claim)
        claims.append(claim)
    return claims


def reconstruction_score(metrics: dict[str, Any]) -> float:
    psnr = float(metrics.get("PSNR", metrics.get("psnr", 0.0)) or 0.0)
    sam = float(metrics.get("SAM", metrics.get("sam", 1.0)) or 1.0)
    ergas = float(metrics.get("ERGAS", metrics.get("ergas", 100.0)) or 100.0)
    return round(psnr - 5.0 * sam - 0.02 * ergas, 6)


def _claim_controlled_tiny(matrix_result: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[str, str, float, dict[str, Any]]:
    text = "controlled chromatic EDOF improves synthetic HSI reconstruction with tiny CNN"
    tiny_rows = [row for row in rows if row.get("dataset") == "synthetic" and row.get("reconstructor") == "tiny_cnn"]
    skipped = [row for row in tiny_rows if row.get("status") == "skipped"]
    available = [row for row in tiny_rows if row.get("status") == "succeeded"]
    controlled = _row(available, "controlled_chromatic_edof")
    conventional = _row(available, "conventional")
    rank_comparison = {"controlled_chromatic_edof": _rank(controlled), "conventional": _rank(conventional)}
    if skipped and not available:
        status, score = "needs_followup", 0.3
    elif controlled and conventional and _rank(controlled) < _rank(conventional):
        status, score = "supported", 0.9
    elif controlled and conventional:
        status, score = "contradicted", 0.8
    else:
        status, score = "needs_followup", 0.4
    return text, status, score, _claim_metadata(matrix_result, rows, ["synthetic"], ["tiny_cnn"], skipped, rank_comparison)


def _claim_achromatic_all(matrix_result: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[str, str, float, dict[str, Any]]:
    text = "achromatic remains best across all reconstructors"
    available_groups = _groups([row for row in rows if row.get("status") == "succeeded"])
    rank_comparison = {}
    achromatic_first = 0
    for key, group in available_groups.items():
        ach = _row(group, "achromatic")
        rank_comparison[str(key)] = {"achromatic_rank": _rank(ach), "best_encoder": min(group, key=lambda item: item.get("rank_within_group") or 999).get("encoder")}
        if ach and _rank(ach) == 1:
            achromatic_first += 1
    if available_groups and achromatic_first == len(available_groups):
        status, score = "supported", 0.9
    elif achromatic_first > 0:
        status, score = "partially_supported", 0.6
    elif available_groups:
        status, score = "contradicted", 0.8
    else:
        status, score = "needs_followup", 0.3
    return text, status, score, _claim_metadata(matrix_result, rows, sorted({r.get("dataset") for r in rows}), sorted({r.get("reconstructor") for r in rows}), [], rank_comparison)


def _claim_chromatic_requires_stronger(matrix_result: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[str, str, float, dict[str, Any]]:
    text = "chromatic coding benefits require stronger reconstruction network"
    succeeded = [row for row in rows if row.get("status") == "succeeded" and row.get("dataset") == "synthetic"]
    linear = [row for row in succeeded if row.get("reconstructor") == "optical_conditioned_linear"]
    tiny = [row for row in succeeded if row.get("reconstructor") == "tiny_cnn"]
    skipped = [row for row in rows if row.get("reconstructor") == "tiny_cnn" and row.get("status") == "skipped"]
    linear_best = _best_encoder(linear)
    tiny_best = _best_encoder(tiny)
    rank_comparison = {"linear_best": linear_best, "tiny_cnn_best": tiny_best}
    if skipped and not tiny:
        status, score = "needs_followup", 0.3
    elif linear_best == "achromatic" and tiny_best in {"chromatic_coded", "controlled_chromatic_edof"}:
        status, score = "supported", 0.9
    elif tiny:
        status, score = "partially_supported", 0.55
    else:
        status, score = "needs_followup", 0.3
    return text, status, score, _claim_metadata(matrix_result, rows, ["synthetic"], ["optical_conditioned_linear", "tiny_cnn"], skipped, rank_comparison)


def _claim_metadata(matrix_result: dict[str, Any], rows: list[dict[str, Any]], datasets: list[Any], reconstructors: list[Any], skipped: list[dict[str, Any]], rank_comparison: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_scope": [item for item in datasets if item],
        "reconstructor_scope": [item for item in reconstructors if item],
        "matrix_evidence": {"matrix_id": matrix_result.get("matrix_id"), "artifact_ids": matrix_result.get("artifact_ids", [])},
        "skipped_conditions": [{"encoder": row.get("encoder"), "reconstructor": row.get("reconstructor"), "error_code": row.get("error_code")} for row in skipped],
        "rank_comparison": rank_comparison,
    }


def _rank_rows(rows: list[dict[str, Any]]) -> None:
    for _, group in _groups([row for row in rows if row.get("status") == "succeeded"]).items():
        for idx, row in enumerate(sorted(group, key=lambda item: -float(item.get("reconstruction_score") or -999999))):
            row["rank_within_group"] = idx + 1


def _groups(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row.get("dataset"), row.get("backend"), row.get("reconstructor"), row.get("forward_mode"))
        grouped.setdefault(key, []).append(row)
    return grouped


def _matrix_summary(matrix_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    best_by_reconstructor: dict[str, dict[str, Any]] = {}
    for reconstructor in sorted({row.get("reconstructor") for row in rows if row.get("reconstructor")}):
        candidates = [row for row in rows if row.get("reconstructor") == reconstructor and row.get("status") == "succeeded"]
        skipped = [row for row in rows if row.get("reconstructor") == reconstructor and row.get("status") == "skipped"]
        if candidates:
            best = min(candidates, key=lambda row: row.get("rank_within_group") or 999)
            best_by_reconstructor[reconstructor] = {"encoder": best["encoder"], "rank": best["rank_within_group"], "score": best["reconstruction_score"]}
        elif skipped:
            best_by_reconstructor[reconstructor] = {"encoder": None, "rank": None, "score": None, "status": "skipped", "error_code": skipped[0].get("error_code")}
    return {
        "matrix_id": matrix_id,
        "row_count": len(rows),
        "succeeded": sum(1 for row in rows if row.get("status") == "succeeded"),
        "skipped": sum(1 for row in rows if row.get("status") == "skipped"),
        "best_by_reconstructor": best_by_reconstructor,
    }


def _matrix_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# HSI Matrix Results",
        "",
        f"Matrix ID: `{result['matrix_id']}`",
        "",
        "| Dataset | Backend | Encoder | Reconstructor | Forward mode | PSNR | SSIM | SAM | ERGAS | Worst-depth SAM | Score | Rank | Evidence | Caveat | Status |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in result["rows"]:
        lines.append(
            "| {dataset} | {backend} | {encoder} | {reconstructor} | {forward} | {psnr} | {ssim} | {sam} | {ergas} | {worst} | {score} | {rank} | {evidence} | {caveat} | {status} |".format(
                dataset=row.get("dataset"),
                backend=row.get("backend"),
                encoder=row.get("encoder"),
                reconstructor=row.get("reconstructor"),
                forward=row.get("forward_mode"),
                psnr=row.get("psnr"),
                ssim=row.get("ssim"),
                sam=row.get("sam"),
                ergas=row.get("ergas"),
                worst=row.get("worst_depth_sam"),
                score=row.get("reconstruction_score"),
                rank=row.get("rank_within_group"),
                evidence=row.get("evidence_level"),
                caveat=row.get("caveat"),
                status=row.get("status"),
            )
        )
    lines.append("")
    lines.append("## Best Encoder By Reconstructor")
    for reconstructor, item in result["summary"]["best_by_reconstructor"].items():
        lines.append(f"- `{reconstructor}`: `{item.get('encoder')}`")
    return "\n".join(lines)


def _skipped_row(dataset: str, backend: str, encoder: str, reconstructor: str, forward_mode: str, error_code: str) -> dict[str, Any]:
    return {
        "status": "skipped",
        "dataset": dataset,
        "backend": backend,
        "encoder": encoder,
        "reconstructor": reconstructor,
        "forward_mode": forward_mode,
        "psnr": None,
        "ssim": None,
        "sam": None,
        "ergas": None,
        "worst_depth_sam": None,
        "reconstruction_score": None,
        "rank_within_group": None,
        "evidence_level": "not_available",
        "caveat": f"skipped: {error_code}",
        "error_code": error_code,
    }


def _caveat(dataset: str, backend: str, reconstructor: str) -> str:
    parts = []
    if dataset == "synthetic":
        parts.append("synthetic dataset")
    else:
        parts.append("local-path public/custom dataset")
    if backend == "mock_deeplens":
        parts.append("mock optical backend, not real camera validation")
    if reconstructor in {"tiny_cnn", "unet_tiny"}:
        parts.append("optional torch reconstructor")
    return "; ".join(parts)


def _matrix_trace(workspace_id: str, matrix_id: str, objective: str, rows: list[dict[str, Any]]) -> MetaTrace:
    now = datetime.now(timezone.utc)
    return MetaTrace(
        trace_id=make_trace_id(workspace_id, matrix_id, "hsi-matrix", "SimulationExperimentalist", f"{objective}:{now.isoformat()}"),
        workspace_id=workspace_id,
        run_id=matrix_id,
        branch_id=None,
        step_id="hsi-matrix",
        actor="SimulationExperimentalist",
        phase="Execute",
        task="run HSI dataset/reconstructor matrix",
        skill_id="hsi-reconstruction",
        skill_version="0.1.0",
        tool="run_hsi_matrix",
        input_refs=[],
        output_refs=[],
        findings=[f"matrix rows={len(rows)} succeeded={sum(1 for row in rows if row.get('status') == 'succeeded')}"],
        limitations=["matrix evidence keeps dataset/backend/reconstructor scope explicit"],
        next_action="review matrix-level claims",
        status="succeeded",
        timestamp_start=now,
        timestamp_end=now,
        parents=[],
        content_hash=None,
        metadata={"objective": objective, "evidence_domain": "hsi_matrix"},
    )


def _row(rows: list[dict[str, Any]], encoder: str) -> dict[str, Any] | None:
    return next((row for row in rows if row.get("encoder") == encoder), None)


def _rank(row: dict[str, Any] | None) -> int | None:
    if row is None:
        return None
    return row.get("rank_within_group")


def _best_encoder(rows: list[dict[str, Any]]) -> str | None:
    if not rows:
        return None
    return min(rows, key=lambda row: row.get("rank_within_group") or 999).get("encoder")
