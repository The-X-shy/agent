"""End-to-end synthetic HSI reconstruction flow with optical-sensitive rendering."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from optiresearch.agents.method_builder import MethodBuilder
from optiresearch.hsi.forward_model import HSIForwardModel
from optiresearch.hsi.optical_features import OpticalFeatureExtractor
from optiresearch.hsi.public_datasets import get_hsi_dataset_adapter
from optiresearch.hsi.reconstruction import run_reconstruction
from optiresearch.memory.claim_evidence import ClaimEvidenceManager
from optiresearch.memory.compiler import MemoryCompiler
from optiresearch.memory.meta_trace import MetaTraceWriter
from optiresearch.memory.schemas import MetaTrace, make_deterministic_id, make_trace_id
from optiresearch.runtime.graph import run_mvp_flow
from optiresearch.schemas.hsi import (
    build_default_hsi_forward_model_spec,
    build_default_hsi_reconstruction_spec,
    build_default_synthetic_hsi_dataset_spec,
)
from optiresearch.storage.file_artifact_store import FileArtifactStore
from optiresearch.storage.sqlite_store import SQLiteStore


def run_hsi_reconstruction_flow(
    objective: str,
    backend: str = "mock_deeplens",
    encoder_type: str = "controlled_chromatic_edof",
    workspace_id: str = "default",
    use_llm: bool = False,
    llm_provider: str | None = None,
    realization: str = "auto",
    forward_mode: str = "depth_spectral_coded",
    reconstructor_type: str = "optical_conditioned_linear",
    dataset_pattern: str = "mixed_materials",
    dataset: str = "synthetic",
    dataset_path: str | None = None,
    reconstructor: str | None = None,
    use_optical_feature_maps: bool = False,
    tiny_cnn_epochs: int = 5,
    tiny_cnn_hidden: int = 32,
    device: str = "cpu",
) -> dict[str, Any]:
    store = SQLiteStore()
    store.init_db()
    artifact_store = FileArtifactStore(store=store)
    selected_reconstructor = reconstructor or reconstructor_type
    run_id = make_deterministic_id(
        "run_hsi",
        workspace_id,
        objective,
        backend,
        encoder_type,
        dataset,
        selected_reconstructor,
        forward_mode,
        use_optical_feature_maps,
    )
    experiment_spec = MethodBuilder().build_mock_optical_spec(objective, encoder_type=encoder_type, backend=backend)
    optical = run_mvp_flow(
        objective,
        workspace_id=workspace_id,
        experiment_spec=experiment_spec,
        backend=backend,
        use_llm=use_llm,
        realization=realization,
    )
    psf_artifact = _find_psf_artifact(artifact_store, optical["run_id"])
    psf_path = artifact_store.resolve_uri(psf_artifact.uri)
    dataset_spec = build_default_synthetic_hsi_dataset_spec(spectral_pattern_type=dataset_pattern)
    if dataset != "synthetic":
        dataset_spec.dataset_family = dataset  # type: ignore[assignment]
        dataset_spec.source = "local" if dataset == "local_npz" else "public_placeholder"
        dataset_spec.dataset_path = dataset_path
    depth_planes = int(optical["run_memory"]["best_metrics"].get("depth_planes", 9))
    forward_spec = build_default_hsi_forward_model_spec(
        optical_artifact_id=psf_artifact.artifact_id,
        psf_cube_uri=psf_artifact.uri,
        depth_planes=depth_planes,
        wavelength_bands=dataset_spec.spectral_bands,
        forward_mode=forward_mode,
    )
    reconstruction_spec = build_default_hsi_reconstruction_spec(
        output_bands=dataset_spec.spectral_bands,
        network_type=selected_reconstructor,
    )
    reconstruction_spec.metadata["use_optical_features"] = bool(use_optical_feature_maps)
    reconstruction_spec.metadata["optical_feature_injection"] = "concat_scalar_maps" if use_optical_feature_maps else "none"
    if use_optical_feature_maps:
        reconstruction_spec.input_channels = 1 + 4
    if selected_reconstructor in {"tiny_cnn", "unet_tiny"}:
        reconstruction_spec.train_config.update({"epochs": tiny_cnn_epochs, "hidden_channels": tiny_cnn_hidden, "device": device})
    root = Path(os.getenv("OPTIRESEARCH_HSI_ROOT", "./workspace/hsi")) / "runs" / run_id
    dataset_dir = root / "dataset"
    reconstruction_dir = root / "reconstruction"
    forward_dir = root / "forward"

    adapter = get_hsi_dataset_adapter(dataset, path=dataset_path, spec=dataset_spec)
    dataset_manifest = adapter.prepare(dataset_dir)
    if dataset_manifest.get("status") == "error":
        return {
            "status": "error",
            "run_id": run_id,
            "optical_run_id": optical["run_id"],
            "error_code": dataset_manifest.get("error_code"),
            "dataset": dataset_manifest,
            "metrics": {},
            "artifact_ids": [],
            "artifact_uris": [],
            "artifact_names": [],
            "claims": [],
            "evidence_level": _evidence_level(backend, optical["run_memory"]["best_metrics"]),
        }
    if "spectral_bands" in dataset_manifest:
        dataset_spec.spectral_bands = int(dataset_manifest["spectral_bands"])
        reconstruction_spec.output_bands = int(dataset_manifest["spectral_bands"])
        forward_spec.wavelength_bands = int(dataset_manifest["spectral_bands"])

    forward_model = HSIForwardModel(forward_spec)
    psf_cube = forward_model.load_psf_cube(str(psf_path))
    optical_features = OpticalFeatureExtractor().extract(psf_cube)

    splits = {}
    for split in ("train", "val", "test"):
        payload = adapter.load_split(split)
        splits[split] = forward_model.render_batch(payload["hsi"], psf_cube, payload.get("depth_indices"), optical_features)

    np.savez_compressed(root / "measurements.npz", train=splits["train"]["measurements"], test=splits["test"]["measurements"])

    coding_weights = None
    if splits["train"]["measurements"].shape[0] > 0:
        _, coding_weights = forward_model.render_measurement_with_coding_weights(
            splits["train"]["targets"][0], psf_cube, int(splits["train"]["depth_indices"][0]), optical_features
        )

    measurement_stats = {
        "train_mean": float(np.mean(splits["train"]["measurements"])),
        "train_std": float(np.std(splits["train"]["measurements"])),
        "test_mean": float(np.mean(splits["test"]["measurements"])),
        "test_std": float(np.std(splits["test"]["measurements"])),
    }
    forward_artifact_paths = forward_model.save_forward_artifacts(
        forward_dir, optical_features, coding_weights, measurement_stats
    )

    reconstruction = run_reconstruction(
        selected_reconstructor,
        splits["train"]["measurements"],
        splits["train"]["targets"],
        splits["test"]["measurements"],
        splits["test"]["targets"],
        splits["test"]["depth_indices"],
        reconstruction_dir,
        optical_features,
        use_optical_feature_maps=use_optical_feature_maps,
        optical_feature_injection="concat_scalar_maps" if use_optical_feature_maps else "none",
        train_options={"epochs": tiny_cnn_epochs, "hidden_channels": tiny_cnn_hidden, "device": device},
    )
    trace_writer = MetaTraceWriter(store)
    trace = trace_writer.write_trace(_hsi_trace(workspace_id, run_id, objective, backend, encoder_type, reconstruction["metrics"], dataset, selected_reconstructor))
    registered = []
    for path in [
        dataset_dir / "dataset_manifest.json",
        root / "measurements.npz",
        *reconstruction["artifacts"],
        *forward_artifact_paths,
    ]:
        artifact_type = _classify_artifact(path.name)
        registered.append(
            artifact_store.register_file(
                path,
                workspace_id=workspace_id,
                run_id=run_id,
                trace_id=trace.trace_id,
                producer="HSIReconstructionPipeline",
                metadata={
                    "filename": path.name,
                    "artifact_type": artifact_type,
                    "backend": backend,
                    "encoder_type": encoder_type,
                    "evidence_domain": "hsi_reconstruction",
                    "forward_mode": forward_mode,
                    "reconstructor_type": selected_reconstructor,
                    "dataset": dataset,
                    "dataset_family": dataset_manifest.get("dataset_family", dataset),
                    "dataset_pattern": dataset_pattern,
                    "use_optical_feature_maps": use_optical_feature_maps,
                },
                metrics=reconstruction["metrics"] if path.name == "reconstruction_metrics.json" else {},
            )
        )

    enhanced_metrics = dict(reconstruction["metrics"])
    enhanced_metrics.update({
        "optical_coding_strength": optical_features.get("coding_strength"),
        "optical_depth_stability_score": optical_features.get("depth_stability_score"),
        "optical_spectral_separability_score": optical_features.get("spectral_separability_score"),
        "optical_band_condition_score": optical_features.get("band_condition_score"),
    })

    run_memory = MemoryCompiler(store=store, artifact_store=artifact_store).compile_run_memory(run_id)
    claims = _review_hsi_claims(store, workspace_id, run_id, backend, encoder_type, dataset, selected_reconstructor)
    return {
        "status": reconstruction.get("status", "succeeded"),
        "run_id": run_id,
        "optical_run_id": optical["run_id"],
        "dataset": dataset_manifest,
        "forward_model": forward_spec.model_dump(mode="json"),
        "reconstruction": reconstruction_spec.model_dump(mode="json"),
        "metrics": enhanced_metrics,
        "optical_features": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in optical_features.items()},
        "artifact_ids": [item.artifact_id for item in registered],
        "artifact_uris": [item.uri for item in registered],
        "artifact_names": [item.metadata.get("filename") for item in registered],
        "run_memory": run_memory.model_dump(mode="json"),
        "claims": [claim.model_dump(mode="json") for claim in claims],
        "evidence_level": _evidence_level(backend, optical["run_memory"]["best_metrics"]),
        "error_code": reconstruction.get("error_code"),
    }


def _find_psf_artifact(artifact_store: FileArtifactStore, run_id: str):
    for artifact in artifact_store.list_artifacts(run_id=run_id):
        if artifact.metadata.get("artifact_type") == "psf_cube" and artifact.metadata.get("filename") == "psf_cube.npz":
            return artifact
    raise ValueError(f"No psf_cube.npz artifact found for run_id={run_id}")


def _classify_artifact(filename: str) -> str:
    if filename == "optical_features.json":
        return "optical_features"
    if filename == "coding_weights.npy":
        return "coding_weights"
    if filename == "measurement_stats.json":
        return "measurement_stats"
    if filename == "forward_model_manifest.json":
        return "forward_manifest"
    if filename == "reconstruction_metrics.json":
        return "metrics"
    if filename == "dataset_manifest.json":
        return "manifest"
    return "hsi_artifact"


def _hsi_trace(workspace_id: str, run_id: str, objective: str, backend: str, encoder_type: str, metrics: dict[str, Any], dataset: str, reconstructor: str) -> MetaTrace:
    now = datetime.now(timezone.utc)
    task = "run HSI reconstruction pipeline"
    return MetaTrace(
        trace_id=make_trace_id(workspace_id, run_id, "hsi-reconstruction", "SimulationExperimentalist", f"{task}:{now.isoformat()}"),
        workspace_id=workspace_id,
        run_id=run_id,
        branch_id=None,
        step_id="hsi-reconstruction",
        actor="SimulationExperimentalist",
        phase="Execute",
        task=task,
        skill_id="hsi-reconstruction",
        skill_version="0.1.0",
        tool="run_hsi_reconstruction_flow",
        input_refs=[],
        output_refs=[],
        findings=[f"HSI reconstruction metrics: {metrics}"],
        limitations=[f"dataset={dataset}", f"reconstructor={reconstructor}", f"backend={backend}"],
        next_action="review reconstruction evidence",
        status="succeeded",
        timestamp_start=now,
        timestamp_end=now,
        parents=[],
        content_hash=None,
        metadata={
            "objective": objective,
            "backend": backend,
            "encoder_type": encoder_type,
            "dataset": dataset,
            "reconstructor": reconstructor,
            "evidence_domain": "hsi_reconstruction",
        },
    )


def _review_hsi_claims(store: SQLiteStore, workspace_id: str, run_id: str, backend: str, encoder_type: str, dataset: str, reconstructor: str):
    manager = ClaimEvidenceManager(store, workspace_id=workspace_id)
    texts = ["HSI reconstruction pipeline is executable end-to-end"]
    if encoder_type == "controlled_chromatic_edof":
        texts.append("controlled chromatic EDOF improves synthetic HSI reconstruction under mock setting")
    claims = []
    for text in texts:
        claim = manager.create_claim(
            text,
            scope={
                "backend": backend,
                "run_id": run_id,
                "evidence_domain": "hsi_reconstruction",
                "encoder_type": encoder_type,
                "dataset": dataset,
                "reconstructor": reconstructor,
                "claim_scope": f"{dataset}/{backend}/{reconstructor}",
            },
        )
        claims.append(manager.review_claim(claim.claim_id))
    return claims


def _evidence_level(backend: str, metrics: dict[str, Any]) -> str:
    if backend == "mock_deeplens":
        return "hsi_reconstruction_mock"
    selected = metrics.get("selected_realization_level") or metrics.get("encoder_behavior_realization_level")
    if selected == "semi_native":
        return "hsi_reconstruction_deeplens_semi_native"
    if selected == "native":
        return "hsi_reconstruction_deeplens_native"
    return "hsi_reconstruction_deeplens_proxy"
