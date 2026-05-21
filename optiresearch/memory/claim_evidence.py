"""Claim-evidence manager for conservative review."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from optiresearch.memory.schemas import ClaimEvidence, EvidenceEdge, make_claim_id
from optiresearch.runtime.backend_metadata import DEEPLENS_PROXY_CAVEAT, DEEPLENS_SMOKE_CAVEAT, backend_metadata
from optiresearch.storage.sqlite_store import SQLiteStore
from optiresearch.llm.registry import get_llm_provider
from optiresearch.llm.structured_output import ClaimReviewDraft


SIMULATION_CAVEAT = "currently simulation-only or mock-backed"
MOCK_CAVEAT = "mock-backed evidence only"


class ClaimEvidenceManager:
    """Create, attach evidence, review, and explain claims."""

    def __init__(self, store: Optional[SQLiteStore] = None, workspace_id: str = "default") -> None:
        self.store = store or SQLiteStore()
        self.store.init_db()
        self.workspace_id = workspace_id

    def create_claim(self, text: str, scope: Optional[dict[str, Any]] = None) -> ClaimEvidence:
        scope = scope or {}
        claim = ClaimEvidence(
            claim_id=make_claim_id(text, scope),
            text=text,
            status="unsupported",
            support_score=0.0,
            support_edges=[],
            contradict_edges=[],
            scope=scope,
            review_status="created",
            required_caveats=self._caveats(scope),
            warnings=[],
            metadata=self._metadata(scope),
        )
        self._save(claim)
        return claim

    def get_claim(self, claim_id: str) -> Optional[ClaimEvidence]:
        payload = self.store.get("claims", claim_id)
        return ClaimEvidence(**payload) if payload else None

    def attach_support(
        self,
        claim_id: str,
        artifact_id: str,
        score: float,
        relation: str = "supports",
    ) -> ClaimEvidence:
        claim = self._require(claim_id)
        claim.support_edges.append(
            EvidenceEdge(
                artifact_id=artifact_id,
                trace_id=None,
                metric_name=None,
                metric_value=None,
                relation="supports" if relation != "qualifies" else "qualifies",
                score=score,
                rationale="manual support edge",
            )
        )
        claim.support_score = max([edge.score for edge in claim.support_edges], default=0.0)
        self._save(claim)
        return claim

    def attach_contradiction(
        self,
        claim_id: str,
        artifact_id: str,
        score: float,
        relation: str = "contradicts",
    ) -> ClaimEvidence:
        claim = self._require(claim_id)
        claim.contradict_edges.append(
            EvidenceEdge(
                artifact_id=artifact_id,
                trace_id=None,
                metric_name=None,
                metric_value=None,
                relation="contradicts",
                score=score,
                rationale=f"manual {relation} edge",
            )
        )
        self._save(claim)
        return claim

    def review_claim(self, claim_id: str) -> ClaimEvidence:
        claim = self._require(claim_id)
        claim.warnings = []
        self._attach_metric_evidence_if_available(claim)
        claim.support_score = max([edge.score for edge in claim.support_edges], default=0.0)
        highest_contradiction = max([edge.score for edge in claim.contradict_edges], default=0.0)
        if claim.contradict_edges and highest_contradiction > claim.support_score:
            claim.status = "contradicted"
        elif not claim.support_edges:
            claim.status = "unsupported"
        elif claim.support_score >= 0.75:
            claim.status = "supported"
        else:
            claim.status = "partially_supported"
        self._downgrade_deeplens_smoke_claim(claim)
        self._apply_public_hsi_rules(claim)
        self._apply_native_optimization_level_rules(claim)
        claim.review_status = "reviewed"
        claim.required_caveats = self._caveats(claim.scope)
        claim.metadata = {**self._metadata(claim.scope), **claim.metadata}
        self._save(claim)
        return claim

    def explain_claim(self, claim_id: str) -> dict[str, Any]:
        claim = self._require(claim_id)
        edges = [*claim.support_edges, *claim.contradict_edges]
        explanation = {
            "claim_id": claim.claim_id,
            "claim_text": claim.text,
            "status": claim.status,
            "support_score": claim.support_score,
            "evidence_table": [
                {
                    "artifact_id": edge.artifact_id,
                    "trace_id": edge.trace_id,
                    "metric_name": edge.metric_name,
                    "metric_value": edge.metric_value,
                    "relation": edge.relation,
                    "score": edge.score,
                    "rationale": edge.rationale,
                }
                for edge in edges
            ],
            "caveats": claim.required_caveats,
            "source_traces": sorted({edge.trace_id for edge in edges if edge.trace_id}),
            "warnings": claim.warnings,
            "evidence_level": claim.metadata.get("evidence_level"),
            "allowed_claim_scope": claim.metadata.get("allowed_claim_scope"),
            "disallowed_claim_scope": claim.metadata.get("disallowed_claim_scope"),
            "required_next_validation": claim.metadata.get("required_next_validation"),
            "compared_baseline": claim.metadata.get("compared_baseline"),
            "compared_metric": claim.metadata.get("compared_metric"),
            "ranking_position": claim.metadata.get("ranking_position"),
            "dataset_scope": claim.metadata.get("dataset_scope"),
            "reconstructor_scope": claim.metadata.get("reconstructor_scope"),
            "matrix_evidence": claim.metadata.get("matrix_evidence"),
            "skipped_conditions": claim.metadata.get("skipped_conditions"),
            "rank_comparison": claim.metadata.get("rank_comparison"),
            "dataset_family": claim.metadata.get("dataset_family"),
            "dataset_manifest_id": claim.metadata.get("dataset_manifest_id"),
            "backend_scope": claim.metadata.get("backend_scope"),
            "real_camera_evidence": claim.metadata.get("real_camera_evidence"),
            "optical_backend_evidence_level": claim.metadata.get("optical_backend_evidence_level"),
            "differentiable": claim.metadata.get("differentiable"),
            "native_parameter_update": claim.metadata.get("native_parameter_update"),
            "gradient_norm": claim.metadata.get("gradient_norm"),
            "parameters_changed": claim.metadata.get("parameters_changed"),
            "loss_before": claim.metadata.get("loss_before"),
            "loss_after": claim.metadata.get("loss_after"),
            "lens_class": claim.metadata.get("lens_class"),
            "realization_level": claim.metadata.get("realization_level"),
            "native_optimization_level": claim.metadata.get("native_optimization_level"),
            "surface_class": claim.metadata.get("surface_class"),
            "lens_file_loaded": claim.metadata.get("lens_file_loaded"),
            "optimizer_step_executed": claim.metadata.get("optimizer_step_executed"),
        }
        return explanation

    def review_claim_with_llm(self, claim_id: str, evidence: dict[str, Any] | None = None, provider=None) -> ClaimReviewDraft:
        claim = self._require(claim_id)
        provider = provider or get_llm_provider()
        if getattr(provider, "available", lambda: False)():
            try:
                return provider.structured_complete(
                    [{"role": "user", "content": f"Claim: {claim.text}\nEvidence: {evidence or {}}"}],
                    ClaimReviewDraft,
                )
            except Exception:
                pass
        return ClaimReviewDraft(
            claim_text=claim.text,
            suggested_status="needs_followup",
            reasoning="Rule fallback: final status is determined by artifact evidence.",
            required_caveats=claim.required_caveats,
            missing_evidence=[],
            follow_up_experiments=[],
            risk_level="medium",
        )

    def list_claims(self) -> list[ClaimEvidence]:
        return [ClaimEvidence(**payload) for payload in self.store.list("claims", workspace_id=self.workspace_id)]

    def _require(self, claim_id: str) -> ClaimEvidence:
        claim = self.get_claim(claim_id)
        if claim is None:
            raise KeyError(f"Unknown claim_id={claim_id}")
        return claim

    def _save(self, claim: ClaimEvidence) -> None:
        self.store.upsert("claims", claim.claim_id, claim, workspace_id=self.workspace_id, run_id=claim.scope.get("run_id"))

    def _caveats(self, scope: dict[str, Any]) -> list[str]:
        scope_text = str(scope).lower()
        caveats: list[str] = []
        if "mock" in scope_text or "simulation" in scope_text:
            caveats.extend([SIMULATION_CAVEAT, MOCK_CAVEAT])
        if scope.get("backend") == "deeplens" and scope.get("backend_capability_level") in {"smoke", "minimal"}:
            caveats.append(DEEPLENS_SMOKE_CAVEAT)
        if scope.get("backend") == "deeplens" and scope.get("encoder_behavior_realization_level") == "adapter_proxy":
            caveats.append(DEEPLENS_PROXY_CAVEAT)
        if scope.get("evidence_domain") == "public_hsi_matrix" and scope.get("backend") == "mock_deeplens":
            caveats.append("public data but synthetic/mock optical measurement")
        return list(dict.fromkeys(caveats))

    def _metadata(self, scope: dict[str, Any]) -> dict[str, Any]:
        backend = str(scope.get("backend", "mock_deeplens"))
        if backend not in {"mock_deeplens", "deeplens"}:
            backend = "mock_deeplens"
        return backend_metadata(
            backend,
            {
                "backend_capability_level": scope.get(
                    "backend_capability_level",
                    "mock" if backend == "mock_deeplens" else "smoke",
                ),
                "encoder_behavior_realized": scope.get(
                    "encoder_behavior_realized",
                    backend == "mock_deeplens",
                ),
                "encoder_behavior_realization_level": scope.get("encoder_behavior_realization_level"),
                "physical_validation_level": scope.get("physical_validation_level"),
                "proxy_transform_applied": scope.get("proxy_transform_applied"),
                "proxy_transform_name": scope.get("proxy_transform_name"),
                "evidence_level": self._evidence_level(scope),
                "allowed_claim_scope": scope.get("claim_scope"),
                "disallowed_claim_scope": "native physical optimization" if scope.get("selected_realization_level") != "native" else None,
                "required_next_validation": "native DeepLens optimization and wavelength-aware HSI validation"
                if scope.get("selected_realization_level") != "native"
                else None,
                "dataset_family": scope.get("dataset_family"),
                "dataset_manifest_id": scope.get("dataset_manifest_id"),
                "backend_scope": scope.get("backend"),
                "real_camera_evidence": bool(scope.get("real_camera_evidence")),
                "optical_backend_evidence_level": self._evidence_level(scope),
                "native_optimization_level": scope.get("native_optimization_level"),
                "surface_class": scope.get("surface_class"),
                "lens_file_loaded": scope.get("lens_file_loaded"),
                "optimizer_step_executed": scope.get("optimizer_step_executed"),
            },
        )

    def _evidence_level(self, scope: dict[str, Any]) -> str:
        if scope.get("evidence_domain") == "deeplens_native_optimization":
            level = scope.get("native_optimization_level")
            if level == "component":
                return "deeplens_native_component_optimization"
            if level == "lens":
                return "deeplens_native_lens_optimization"
            if level == "optical_hsi_codesign":
                return "deeplens_native_optical_hsi_codesign"
            return "deeplens_native_optimization_unqualified"
        if scope.get("evidence_domain") == "native_optimization_probe":
            if scope.get("realization_level") == "native" and scope.get("differentiable"):
                return "deeplens_native_optimization"
            if scope.get("realization_level") == "semi_native":
                return "deeplens_semi_native"
            return "deeplens_native_optimization"
        if scope.get("evidence_domain") == "public_hsi_matrix":
            dataset_family = scope.get("dataset_family")
            if dataset_family == "synthetic":
                return "synthetic_hsi"
            if scope.get("backend") == "mock_deeplens":
                return "public_hsi_mock"
            level = scope.get("selected_realization_level") or scope.get("encoder_behavior_realization_level")
            if level == "native":
                return "public_hsi_deeplens_native"
            if level == "semi_native":
                return "public_hsi_deeplens_semi_native"
            return "public_hsi_deeplens_proxy"
        if scope.get("backend") == "mock_deeplens":
            return "hsi_reconstruction_mock" if scope.get("evidence_domain") == "hsi_reconstruction" else "mock"
        if scope.get("backend") != "deeplens":
            return "real_lab" if scope.get("backend") == "real_lab" else "mock"
        if scope.get("evidence_domain") == "hsi_reconstruction":
            level = scope.get("selected_realization_level") or scope.get("encoder_behavior_realization_level")
            if level == "native":
                return "hsi_reconstruction_deeplens_native"
            if level == "semi_native":
                return "hsi_reconstruction_deeplens_semi_native"
            return "hsi_reconstruction_deeplens_proxy"
        level = scope.get("selected_realization_level") or scope.get("encoder_behavior_realization_level")
        if level == "native":
            return "deeplens_native"
        if level == "semi_native":
            return "deeplens_semi_native"
        if level == "adapter_proxy":
            return "deeplens_adapter_proxy"
        if scope.get("backend_capability_level") in {"smoke", "minimal"}:
            return "deeplens_smoke"
        return "deeplens_adapter_proxy"

    def _downgrade_deeplens_smoke_claim(self, claim: ClaimEvidence) -> None:
        scope = claim.scope
        if scope.get("backend") != "deeplens":
            return
        lower = claim.text.lower()
        if scope.get("backend_capability_level") in {"smoke", "minimal"}:
            self._downgrade_smoke_claim(claim, lower)
        elif scope.get("encoder_behavior_realization_level") == "adapter_proxy":
            self._downgrade_proxy_claim(claim, lower)
        elif scope.get("selected_realization_level") == "semi_native":
            self._downgrade_semi_native_claim(claim, lower)
        self._apply_hsi_reconstruction_rules(claim, lower)

    def _downgrade_smoke_claim(self, claim: ClaimEvidence, lower: str) -> None:
        if "valid psf artifact" in lower or "valid psf artifacts" in lower:
            return
        encoder_claim = any(
            keyword in lower
            for keyword in (
                "improves",
                "improve",
                "best",
                "better",
                "encoder",
                "depth stability",
                "spectral separability",
                "controlled chromatic edof",
            )
        )
        if encoder_claim and claim.status == "supported":
            claim.status = "partially_supported"
            claim.warnings.append("deeplens_smoke_claim_downgraded")
        if "controlled chromatic edof is best under real deeplens" in lower and claim.scope.get("encoder_behavior_realized") is False:
            claim.status = "unsupported"
            claim.warnings.append("encoder_behavior_not_realized")

    def _downgrade_proxy_claim(self, claim: ClaimEvidence, lower: str) -> None:
        if "adapter-proxy" in lower or "adapter proxy" in lower:
            return
        if "encoder-specific baseline artifact" in lower or "encoder specific baseline artifact" in lower:
            return
        if "valid psf artifact" in lower or "valid psf artifacts" in lower:
            return
        physical_claim = any(
            keyword in lower
            for keyword in (
                "physically validated",
                "physical",
                "native",
                "real optical",
                "under real deeplens",
            )
        )
        if physical_claim and claim.status == "supported":
            claim.status = "partially_supported"
            claim.warnings.append("deeplens_proxy_physical_claim_downgraded")

    def _downgrade_semi_native_claim(self, claim: ClaimEvidence, lower: str) -> None:
        normalized = lower.replace("semi-native", "").replace("semi native", "")
        if "native" in normalized or "physically optimized" in normalized or "physical optimization" in normalized:
            claim.status = "needs_followup"
            claim.warnings.append("semi_native_native_claim_needs_followup")
            return
        if claim.scope.get("semi_native_succeeded") is not True and claim.status == "supported":
            claim.status = "partially_supported"
            claim.warnings.append("semi_native_not_confirmed")

    def _attach_metric_evidence_if_available(self, claim: ClaimEvidence) -> None:
        metric_name, threshold = self._metric_rule(claim.text)
        if metric_name is None:
            return
        if any(edge.metric_name == metric_name for edge in [*claim.support_edges, *claim.contradict_edges]):
            return
        artifact = self._find_metric_artifact(claim.scope.get("run_id"), metric_name)
        if artifact is None:
            claim.warnings.append(f"missing_evidence: {metric_name}")
            return
        metric_value = artifact["metrics"].get(metric_name)
        relation = "supports" if isinstance(metric_value, (int, float)) and metric_value >= threshold else "qualifies"
        score = min(0.95, float(metric_value)) if isinstance(metric_value, (int, float)) else 0.0
        if relation == "qualifies":
            score = min(score, 0.74)
        claim.support_edges.append(
            EvidenceEdge(
                artifact_id=artifact["artifact_id"],
                trace_id=artifact.get("trace_id"),
                metric_name=metric_name,
                metric_value=metric_value,
                relation=relation,
                score=score,
                rationale=f"{metric_name}={metric_value} compared with threshold {threshold}",
            )
        )

    def _metric_rule(self, text: str) -> tuple[Optional[str], float]:
        lower = text.lower()
        if "depth stability" in lower:
            return "psf_depth_similarity", 0.8
        if "spectral separability" in lower:
            return "spectral_separability", 0.3
        if "hsi reconstruction" in lower or "reconstruction pipeline" in lower:
            return "PSNR", 1.0
        return None, 0.0

    def _find_metric_artifact(self, run_id: Any, metric_name: str) -> Optional[dict[str, Any]]:
        artifacts = self.store.list("artifacts", run_id=run_id) if run_id else self.store.list("artifacts")
        for artifact in artifacts:
            if metric_name in artifact.get("metrics", {}):
                return artifact
        return None

    def _apply_native_optimization_level_rules(self, claim: ClaimEvidence) -> None:
        if claim.scope.get("evidence_domain") != "deeplens_native_optimization":
            return

        lower = claim.text.lower()
        scope = claim.scope
        level = scope.get("native_optimization_level")
        has_component_chain = all(
            [
                scope.get("surface_probe_succeeded") is True,
                scope.get("requires_grad_true") is True,
                _positive(scope.get("gradient_norm")),
                scope.get("parameters_changed") is True,
                scope.get("optimizer_step_executed") is True,
            ]
        )
        has_lens_chain = all(
            [
                scope.get("lens_file_loaded") is True,
                scope.get("lens_psf_backward_success") is True,
                _positive(scope.get("gradient_norm")),
                scope.get("parameters_changed") is True,
                scope.get("optimizer_step_executed") is True,
            ]
        )
        has_hsi_chain = all(
            [
                has_lens_chain,
                scope.get("deeplens_psf_feeds_hsi_loss") is True,
                scope.get("hsi_loss_backward_reaches_optical_parameter") is True,
                scope.get("hsi_metric_improved") is True,
            ]
        )

        claim.metadata["native_optimization_level"] = level
        claim.metadata["surface_class"] = scope.get("surface_class")
        claim.metadata["lens_file_loaded"] = scope.get("lens_file_loaded")
        claim.metadata["optimizer_step_executed"] = scope.get("optimizer_step_executed")
        claim.metadata["evidence_level"] = self._evidence_level(scope)

        # HSI proxy co-design: requires component chain + HSI loss backward
        has_hsi_proxy_chain = all(
            [
                has_component_chain,
                scope.get("hsi_loss_after") is not None,
                _positive(scope.get("gradient_norm")),
            ]
        )

        if "optical-hsi" in lower or "optical hsi" in lower or "co-design" in lower or "hsi reconstruction" in lower:
            if "proxy" in lower or "proxy co-design" in lower:
                if has_hsi_proxy_chain and claim.support_edges:
                    claim.status = "supported"
                    claim.support_score = max(claim.support_score, 0.80)
                    claim.metadata["evidence_level"] = "native_hsi_proxy"
                else:
                    claim.status = "needs_followup"
                    claim.warnings.append("native_hsi_proxy_requires_component_chain_and_hsi_loss_backward")
            elif "lens simulation" in lower:
                has_lens_sim_chain = all([
                    has_hsi_proxy_chain,
                    scope.get("phase_to_fft_proxy_used") is False,
                    scope.get("full_wave_optics", True) is False,
                    scope.get("deeplens_native_psf_path") is not None,
                ])
                if has_lens_sim_chain and claim.support_edges:
                    claim.status = "supported"
                    claim.support_score = max(claim.support_score, 0.80)
                    claim.metadata["evidence_level"] = "native_lens_simulation"
                else:
                    claim.status = "needs_followup"
                    claim.warnings.append("native_lens_simulation_requires_deeplens_geometric_psf")
            elif "full wave-optics" in lower:
                claim.status = "needs_followup"
                claim.warnings.append("full_waveoptics_requires_differentiable_coherent_asm_path")
            elif "full" in lower or ("reconstruction" in lower and "real" not in lower and "proxy" not in lower):
                has_full_recon_chain = all([
                    has_hsi_proxy_chain,
                    scope.get("full_reconstruction_loss_used") is True,
                    scope.get("recon_gradient_norm", 0) > 0,
                    scope.get("phase_to_fft_proxy_used") is True,
                ])
                if has_full_recon_chain and claim.support_edges:
                    claim.status = "supported"
                    claim.support_score = max(claim.support_score, 0.85)
                    claim.metadata["evidence_level"] = "native_full_reconstruction_proxy"
                else:
                    claim.status = "needs_followup"
                    claim.warnings.append("full_native_hsi_reconstruction_requires_reconstructor_gradient")
            elif "real hsi" in lower or "real camera" in lower:
                claim.status = "unsupported"
                claim.warnings.append("real_hsi_requires_real_camera_validation")
            else:
                if level != "optical_hsi_codesign" or not has_hsi_chain:
                    claim.status = "needs_followup"
                    claim.warnings.append("native_optical_hsi_codesign_requires_hsi_loss")
            return

        if "lens optimization" in lower or "differentiable lens" in lower:
            if level != "lens" or not has_lens_chain:
                claim.status = "needs_followup"
                claim.warnings.append("native_lens_optimization_requires_lensfile_psf_backward")
            return

        if "component optimization" in lower:
            if has_component_chain and claim.support_edges:
                claim.status = "supported"
                claim.support_score = max(claim.support_score, 0.85)
            else:
                claim.status = "unsupported"
                claim.warnings.append("native_component_optimization_probe_incomplete")

    def _apply_public_hsi_rules(self, claim: ClaimEvidence) -> None:
        if claim.scope.get("evidence_domain") != "public_hsi_matrix":
            return
        lower = claim.text.lower()
        scope = claim.scope
        dataset_family = scope.get("dataset_family")
        claim.metadata["dataset_family"] = dataset_family
        claim.metadata["dataset_manifest_id"] = scope.get("dataset_manifest_id")
        claim.metadata["backend_scope"] = scope.get("backend")
        claim.metadata["real_camera_evidence"] = bool(scope.get("real_camera_evidence"))
        claim.metadata["optical_backend_evidence_level"] = self._evidence_level(scope)

        if "real camera hsi" in lower:
            claim.status = "unsupported"
            claim.warnings.append("real_camera_evidence_missing")
            return

        if "synthetic hsi" in lower and dataset_family != "synthetic":
            claim.status = "unsupported"
            claim.warnings.append("claim_scope_dataset_mismatch")
            return

        if "public dataset result validates optical design" in lower:
            if scope.get("backend") == "mock_deeplens":
                claim.status = "partially_supported"
                claim.support_score = max(claim.support_score, 0.55)
                claim.warnings.append("public_data_but_mock_optical_measurement")
                return
            if scope.get("selected_realization_level") != "native":
                claim.status = "partially_supported"
                claim.warnings.append("public_data_but_non_native_optical_backend")
                return

        if "controlled chromatic edof improves public hsi reconstruction" in lower:
            required = [dataset_family and dataset_family != "synthetic", scope.get("dataset_manifest_id"), scope.get("backend"), scope.get("reconstructor") or scope.get("matrix_result")]
            if not all(required):
                claim.status = "needs_followup"
                claim.warnings.append("public_hsi_scope_incomplete")
                return
            rows = (scope.get("matrix_result") or {}).get("rows", [])
            controlled = next((row for row in rows if row.get("encoder") == "controlled_chromatic_edof"), None)
            conventional = next((row for row in rows if row.get("encoder") == "conventional"), None)
            if controlled is None or conventional is None:
                claim.status = "needs_followup"
                claim.warnings.append("public_matrix_baseline_missing")
                return
            claim.metadata["rank_comparison"] = {
                "controlled_chromatic_edof": controlled.get("rank_within_group"),
                "conventional": conventional.get("rank_within_group"),
            }
            if (controlled.get("rank_within_group") or 999) < (conventional.get("rank_within_group") or 999):
                claim.status = "supported"
                claim.support_score = max(claim.support_score, 0.85)
            else:
                claim.status = "contradicted"
                claim.warnings.append("controlled_edof_not_better_than_public_baseline")

    def _apply_hsi_reconstruction_rules(self, claim: ClaimEvidence, lower: str) -> None:
        if claim.scope.get("evidence_domain") != "hsi_reconstruction":
            return

        if "real hsi reconstruction" in lower and claim.scope.get("selected_realization_level") != "native":
            claim.status = "needs_followup"
            claim.warnings.append("native_hsi_pipeline_required")
            return

        if "best for real hsi reconstruction" in lower:
            claim.status = "needs_followup"
            claim.warnings.append("real_hsi_requires_native_validation")
            return

        if "all encoders perform identically" in lower:
            self._review_identical_encoders_claim(claim)
            return

        has_reconstruction_edge = any(edge.metric_name in {"PSNR", "SAM"} for edge in claim.support_edges)
        if "pipeline is executable end-to-end" in lower and has_reconstruction_edge:
            claim.status = "supported"
            claim.metadata["evidence_level"] = claim.metadata.get("evidence_level", "hsi_reconstruction_mock")
        elif "controlled chromatic edof improves hsi reconstruction under mock setting" in lower and claim.scope.get("backend") == "mock_deeplens" and has_reconstruction_edge:
            self._review_ranking_claim(claim)
        elif not has_reconstruction_edge:
            claim.status = "unsupported"
            claim.warnings.append("missing_reconstruction_metrics")

    def _review_ranking_claim(self, claim: ClaimEvidence) -> None:
        baseline_data = self._find_baseline_comparison(claim.scope.get("run_id"))
        if baseline_data is None:
            claim.status = "needs_followup"
            claim.warnings.append("baseline_comparison_not_found")
            return
        runs = baseline_data.get("runs", [])
        controlled = next((r for r in runs if r.get("encoder_type") == "controlled_chromatic_edof"), None)
        conventional = next((r for r in runs if r.get("encoder_type") == "conventional"), None)
        if controlled is None or conventional is None:
            claim.status = "needs_followup"
            claim.warnings.append("baseline_data_incomplete")
            return
        controlled_score = float(controlled.get("reconstruction_score", 0))
        conventional_score = float(conventional.get("reconstruction_score", 0))
        sorted_runs = sorted(runs, key=lambda r: -float(r.get("reconstruction_score", 0)))
        ranking_position = next((i + 1 for i, r in enumerate(sorted_runs) if r.get("encoder_type") == "controlled_chromatic_edof"), len(runs))
        claim.metadata["compared_baseline"] = "conventional"
        claim.metadata["compared_metric"] = "reconstruction_score"
        claim.metadata["ranking_position"] = ranking_position
        if controlled_score > conventional_score:
            claim.status = "supported"
            claim.warnings.append("synthetic_mock_backed_evidence_only")
        else:
            claim.status = "contradicted"
            claim.warnings.append("controlled_edof_not_better_than_conventional")

    def _review_identical_encoders_claim(self, claim: ClaimEvidence) -> None:
        baseline_data = self._find_baseline_comparison(claim.scope.get("run_id"))
        if baseline_data is None:
            claim.status = "unsupported"
            claim.warnings.append("missing_baseline_comparison")
            return
        runs = baseline_data.get("runs", [])
        sam_values = {r.get("SAM") for r in runs if r.get("SAM") is not None}
        psnr_values = {r.get("PSNR") for r in runs if r.get("PSNR") is not None}
        if len(sam_values) > 1 or len(psnr_values) > 1:
            claim.status = "contradicted"
            claim.warnings.append("encoder_reconstruction_metrics_differ")
            return
        claim.status = "supported"

    def _find_baseline_comparison(self, run_id: Any) -> dict | None:
        baselines_root = Path(os.getenv("OPTIRESEARCH_HSI_BASELINE_ROOT", "./workspace/hsi/baselines"))
        for backend_dir in baselines_root.iterdir() if baselines_root.exists() else []:
            if not backend_dir.is_dir():
                continue
            comparison_path = backend_dir / "hsi_baseline_comparison.json"
            if comparison_path.exists():
                try:
                    return json.loads(comparison_path.read_text(encoding="utf-8"))
                except Exception:
                    pass
        return None


def _positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and float(value) > 0.0
