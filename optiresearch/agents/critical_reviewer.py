"""Rule-based Critical Reviewer."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from optiresearch.memory.claim_evidence import ClaimEvidenceManager
from optiresearch.memory.meta_trace import MetaTraceWriter
from optiresearch.memory.run_memory import RunMemoryStore
from optiresearch.memory.schemas import ClaimEvidence, MetaTrace, make_trace_id
from optiresearch.storage.file_artifact_store import FileArtifactStore
from optiresearch.storage.sqlite_store import SQLiteStore


class CriticalReviewer:
    """Generate conservative metric claims and review artifact support."""

    def __init__(self, store: SQLiteStore, artifact_store: FileArtifactStore, workspace_id: str) -> None:
        self.store = store
        self.artifact_store = artifact_store
        self.workspace_id = workspace_id
        self.claims = ClaimEvidenceManager(store, workspace_id=workspace_id)
        self.run_memories = RunMemoryStore(store)

    def review_claims(self, run_id: str, trace_writer: MetaTraceWriter | None = None) -> list[ClaimEvidence]:
        memory = self.run_memories.latest(run_id)
        metrics = memory.best_metrics if memory else {}
        backend = self._backend_for_run(run_id)
        meta = self._backend_metadata_for_run(run_id, backend)
        claim_prefix = "The mock optical encoder" if backend == "mock_deeplens" else "The DeepLens optical encoder"
        claim_specs = [
            f"{claim_prefix} provides measurable depth stability.",
            f"{claim_prefix} provides measurable spectral separability.",
        ]
        if backend == "deeplens" and self._metrics_artifact(run_id) is not None:
            claim_specs.append("DeepLens backend produced valid PSF artifacts.")
            if meta.get("encoder_behavior_realization_level") == "adapter_proxy":
                claim_specs.append("DeepLens adapter can produce encoder-specific baseline artifacts.")
        reviewed: list[ClaimEvidence] = []
        for text in claim_specs:
            claim = self.claims.create_claim(
                text,
                scope={"backend": backend, "run_id": run_id, **meta},
            )
            if "valid PSF artifacts" in text:
                artifact = self._metrics_artifact(run_id)
                if artifact is not None:
                    self.claims.attach_support(claim.claim_id, artifact.artifact_id, 0.9)
            reviewed.append(self.claims.review_claim(claim.claim_id))
        if trace_writer:
            trace_writer.write_trace(self._review_trace(run_id, reviewed, backend))
        return reviewed

    def _metrics_artifact(self, run_id: str) -> Any | None:
        for artifact in self.artifact_store.list_artifacts(run_id=run_id):
            if artifact.metadata.get("filename") == "optical_metrics.json":
                return artifact
        return None

    def _backend_for_run(self, run_id: str) -> str:
        for trace in reversed(self.store.list("meta_traces", run_id=run_id)):
            backend = trace.get("metadata", {}).get("backend")
            if backend:
                return str(backend)
        return "mock_deeplens"

    def _backend_metadata_for_run(self, run_id: str, backend: str) -> dict[str, Any]:
        for artifact in self.artifact_store.list_artifacts(run_id=run_id):
            if artifact.metrics:
                return {
                    "backend_capability_level": artifact.metadata.get(
                        "backend_capability_level",
                        artifact.metrics.get("backend_capability_level", "mock" if backend == "mock_deeplens" else "smoke"),
                    ),
                    "encoder_behavior_realized": artifact.metadata.get(
                        "encoder_behavior_realized",
                        artifact.metrics.get("encoder_behavior_realized", backend == "mock_deeplens"),
                    ),
                    "encoder_behavior_realization_level": artifact.metadata.get(
                        "encoder_behavior_realization_level",
                        artifact.metrics.get("encoder_behavior_realization_level"),
                    ),
                    "physical_validation_level": artifact.metadata.get(
                        "physical_validation_level",
                        artifact.metrics.get("physical_validation_level"),
                    ),
                    "proxy_transform_applied": artifact.metadata.get(
                        "proxy_transform_applied",
                        artifact.metrics.get("proxy_transform_applied"),
                    ),
                    "proxy_transform_name": artifact.metadata.get(
                        "proxy_transform_name",
                        artifact.metrics.get("proxy_transform_name"),
                    ),
                    "selected_realization_level": artifact.metadata.get(
                        "selected_realization_level",
                        artifact.metrics.get("selected_realization_level"),
                    ),
                    "semi_native_succeeded": artifact.metadata.get(
                        "semi_native_succeeded",
                        artifact.metrics.get("semi_native_succeeded"),
                    ),
                    "claim_scope": artifact.metadata.get(
                        "claim_scope",
                        artifact.metrics.get("claim_scope"),
                    ),
                }
        return {
            "backend_capability_level": "mock" if backend == "mock_deeplens" else "smoke",
            "encoder_behavior_realized": backend == "mock_deeplens",
        }

    def _review_trace(self, run_id: str, claims: list[ClaimEvidence], backend: str) -> MetaTrace:
        task = f"review {backend} simulation claims against artifacts"
        now = datetime.now(timezone.utc)
        return MetaTrace(
            trace_id=make_trace_id(self.workspace_id, run_id, "review", "CriticalReviewer", task),
            workspace_id=self.workspace_id,
            run_id=run_id,
            branch_id=None,
            step_id="review",
            actor="CriticalReviewer",
            phase="Review",
            task=task,
            skill_id="evidence-review",
            skill_version="0.1.0",
            tool="ClaimEvidenceManager.review_claim",
            input_refs=[],
            output_refs=[claim.claim_id for claim in claims],
            findings=[f"claim {claim.claim_id}: {claim.status}" for claim in claims],
            limitations=[f"{backend} evidence requires backend-specific caveats"],
            next_action="query memory for evidence context",
            status="succeeded",
            timestamp_start=now,
            timestamp_end=now,
            parents=[],
            content_hash=None,
            metadata={"backend": backend},
        )
