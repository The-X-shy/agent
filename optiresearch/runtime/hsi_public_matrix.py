"""Public/local HSI matrix protocol with structured skips."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from optiresearch.adapters.deeplens import DeepLensAdapter
from optiresearch.hsi.public_datasets import get_hsi_dataset_adapter
from optiresearch.memory.claim_evidence import ClaimEvidenceManager
from optiresearch.runtime.baselines import ENCODER_TYPES
from optiresearch.runtime.hsi_matrix import reconstruction_score
from optiresearch.runtime.hsi_pipeline import run_hsi_reconstruction_flow
from optiresearch.storage.file_artifact_store import FileArtifactStore
from optiresearch.storage.sqlite_store import SQLiteStore


def run_public_hsi_matrix(
    dataset: str,
    dataset_path: str | None = None,
    backend: str = "mock_deeplens",
    encoders: list[str] | None = None,
    reconstructors: list[str] | None = None,
    forward_modes: list[str] | None = None,
    realization: str = "auto",
    workspace_id: str = "default",
) -> dict[str, Any]:
    encoders = encoders or ENCODER_TYPES
    reconstructors = reconstructors or ["optical_conditioned_linear"]
    forward_modes = forward_modes or ["depth_spectral_coded"]
    matrix_id = _matrix_id(dataset, dataset_path, backend, encoders, reconstructors, forward_modes, realization)
    root = Path(os.getenv("OPTIRESEARCH_HSI_ROOT", "./workspace/hsi")) / "public_matrix" / matrix_id
    root.mkdir(parents=True, exist_ok=True)

    if backend == "deeplens":
        env = DeepLensAdapter().validate_environment()
        if not env["available"]:
            return _write_skip(root, matrix_id, dataset, backend, "DEEPLENS_NOT_AVAILABLE", env.get("message", "DeepLens unavailable"))

    adapter = get_hsi_dataset_adapter(dataset, path=dataset_path)
    prepared = adapter.prepare(root / "dataset")
    if prepared.get("status") in {"error", "skipped"}:
        return _write_skip(root, matrix_id, dataset, backend, prepared.get("error_code", "DATASET_UNAVAILABLE"), prepared.get("message", "Dataset unavailable"), prepared)

    rows: list[dict[str, Any]] = []
    for encoder in encoders:
        for reconstructor in reconstructors:
            for forward_mode in forward_modes:
                result = run_hsi_reconstruction_flow(
                    f"Public HSI matrix [{dataset}/{backend}/{encoder}/{reconstructor}/{forward_mode}]",
                    backend=backend,
                    encoder_type=encoder,
                    workspace_id=workspace_id,
                    dataset=dataset,
                    dataset_path=dataset_path,
                    reconstructor=reconstructor,
                    forward_mode=forward_mode,
                    realization=realization,
                )
                if result.get("status") in {"error", "skipped"}:
                    rows.append(
                        _row_base(dataset, prepared, backend, realization, encoder, reconstructor, forward_mode)
                        | {"status": "skipped", "error_code": result.get("error_code", "RUN_SKIPPED"), "caveat": result.get("error_code", "RUN_SKIPPED")}
                    )
                    continue
                metrics = result["metrics"]
                rows.append(
                    _row_base(dataset, prepared, backend, realization, encoder, reconstructor, forward_mode)
                    | {
                        "status": "succeeded",
                        "run_id": result["run_id"],
                        "metrics": metrics,
                        "psnr": metrics.get("PSNR"),
                        "ssim": metrics.get("SSIM"),
                        "sam": metrics.get("SAM"),
                        "ergas": metrics.get("ERGAS"),
                        "worst_depth_sam": metrics.get("worst_depth_SAM"),
                        "reconstruction_score": reconstruction_score(metrics),
                        "evidence_level": _public_evidence_level(prepared.get("dataset_family"), backend, realization),
                        "caveat": _public_caveat(prepared.get("dataset_family"), backend, realization),
                    }
                )
    _rank_rows(rows)
    summary = {
        "matrix_id": matrix_id,
        "dataset": dataset,
        "dataset_family": prepared.get("dataset_family"),
        "backend": backend,
        "row_count": len(rows),
        "succeeded": sum(1 for row in rows if row.get("status") == "succeeded"),
        "skipped": sum(1 for row in rows if row.get("status") == "skipped"),
    }
    payload = {"status": "succeeded", "matrix_id": matrix_id, "dataset_manifest": prepared, "rows": rows, "summary": summary}
    _write_outputs(root, payload)
    _register_outputs(root, matrix_id, workspace_id)
    _create_public_claims(payload, workspace_id)
    return payload


def _row_base(dataset: str, manifest: dict[str, Any], backend: str, realization: str, encoder: str, reconstructor: str, forward_mode: str) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "dataset_family": manifest.get("dataset_family", dataset),
        "backend": backend,
        "realization_level": realization,
        "encoder": encoder,
        "reconstructor": reconstructor,
        "forward_mode": forward_mode,
        "dataset_manifest_id": manifest.get("dataset_id"),
    }


def _rank_rows(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") == "succeeded":
            key = (row.get("dataset"), row.get("backend"), row.get("reconstructor"), row.get("forward_mode"))
            groups.setdefault(key, []).append(row)
    for group in groups.values():
        for idx, row in enumerate(sorted(group, key=lambda item: -float(item.get("reconstruction_score") or -999999))):
            row["rank_within_group"] = idx + 1


def _write_skip(root: Path, matrix_id: str, dataset: str, backend: str, code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    rows = [{"status": "skipped", "dataset": dataset, "backend": backend, "error_code": code, "message": message, "details": details or {}}]
    payload = {
        "status": "skipped",
        "matrix_id": matrix_id,
        "error_code": code,
        "message": message,
        "rows": rows,
        "summary": {"matrix_id": matrix_id, "row_count": len(rows), "succeeded": 0, "skipped": len(rows)},
    }
    _write_outputs(root, payload)
    return payload


def _write_outputs(root: Path, payload: dict[str, Any]) -> None:
    (root / "public_hsi_matrix_results.json").write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    (root / "public_hsi_matrix_summary.json").write_text(json.dumps(payload["summary"], ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    lines = [
        "# Public HSI Matrix Results",
        "",
        f"Matrix ID: `{payload['matrix_id']}`",
        "",
        "| Dataset | Backend | Encoder | Reconstructor | PSNR | SAM | Score | Rank | Evidence | Status | Caveat |",
        "|---|---|---|---|---:|---:|---:|---:|---|---|---|",
    ]
    for row in payload.get("rows", []):
        lines.append(
            f"| {row.get('dataset')} | {row.get('backend')} | {row.get('encoder')} | {row.get('reconstructor')} | {row.get('psnr')} | {row.get('sam')} | {row.get('reconstruction_score')} | {row.get('rank_within_group')} | {row.get('evidence_level')} | {row.get('status')} | {row.get('caveat') or row.get('message')} |"
        )
    (root / "public_hsi_matrix_results.md").write_text("\n".join(lines), encoding="utf-8")


def _register_outputs(root: Path, matrix_id: str, workspace_id: str) -> None:
    store = SQLiteStore()
    store.init_db()
    artifact_store = FileArtifactStore(store=store)
    for name in ("public_hsi_matrix_results.json", "public_hsi_matrix_results.md", "public_hsi_matrix_summary.json"):
        artifact_store.register_file(
            root / name,
            workspace_id=workspace_id,
            run_id=matrix_id,
            trace_id=None,
            producer="PublicHSIMatrixRunner",
            metadata={"filename": name, "artifact_type": "public_hsi_matrix", "evidence_domain": "public_hsi_matrix"},
            metrics={},
        )


def _create_public_claims(payload: dict[str, Any], workspace_id: str) -> None:
    if payload.get("status") != "succeeded":
        return
    manager = ClaimEvidenceManager(workspace_id=workspace_id)
    manifest = payload.get("dataset_manifest", {})
    claim = manager.create_claim(
        "public dataset result validates optical design",
        scope={
            "evidence_domain": "public_hsi_matrix",
            "dataset_family": manifest.get("dataset_family"),
            "dataset_manifest_id": manifest.get("dataset_id"),
            "backend": payload["summary"].get("backend"),
            "matrix_id": payload.get("matrix_id"),
        },
    )
    manager.review_claim(claim.claim_id)


def _public_evidence_level(dataset_family: str | None, backend: str, realization: str) -> str:
    if dataset_family == "synthetic":
        return "synthetic_hsi"
    if backend == "mock_deeplens":
        return "public_hsi_mock"
    if realization == "semi_native":
        return "public_hsi_deeplens_semi_native"
    if realization == "native":
        return "public_hsi_deeplens_native"
    return "public_hsi_deeplens_proxy"


def _public_caveat(dataset_family: str | None, backend: str, realization: str) -> str:
    if dataset_family == "synthetic":
        return "synthetic dataset, not public HSI performance"
    if backend == "mock_deeplens":
        return "public/local data but synthetic/mock optical measurement"
    if realization != "native":
        return "public/local data with DeepLens non-native or proxy optical backend"
    return "public/local dataset with native DeepLens scope; still not real camera validation"


def _matrix_id(dataset: str, dataset_path: str | None, backend: str, encoders: list[str], reconstructors: list[str], forward_modes: list[str], realization: str) -> str:
    from optiresearch.memory.schemas import make_deterministic_id

    return make_deterministic_id("public_hsi_matrix", dataset, dataset_path, backend, encoders, reconstructors, forward_modes, realization)
