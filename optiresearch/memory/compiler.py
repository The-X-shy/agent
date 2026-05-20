"""Compile immutable traces and artifacts into RunMemory projections."""

from __future__ import annotations

import json
from typing import Any

from optiresearch.memory.schemas import RunMemory
from optiresearch.runtime.backend_metadata import backend_metadata
from optiresearch.storage.file_artifact_store import FileArtifactStore
from optiresearch.storage.sqlite_store import SQLiteStore


DECISION_KEYWORDS = ("decision", "decided", "选择", "转向", "改为")
SCALAR_METRIC_KEYS = {
    "encoder_type",
    "backend_capability_level",
    "encoder_behavior_realized",
    "encoder_behavior_realization_level",
    "physical_validation_level",
    "proxy_transform_applied",
    "proxy_transform_name",
    "selected_realization_level",
    "semi_native_attempted",
    "semi_native_succeeded",
    "proxy_fallback_used",
    "claim_scope",
    "deeplens_mtf_mean",
    "deeplens_energy_efficiency",
}


class MemoryCompiler:
    """Build a compact run summary from trace and artifact records."""

    def __init__(self, store: SQLiteStore | None = None, artifact_store: FileArtifactStore | None = None) -> None:
        self.store = store or SQLiteStore()
        self.store.init_db()
        self.artifact_store = artifact_store or FileArtifactStore(store=self.store)

    def compile_run_memory(self, run_id: str) -> RunMemory:
        traces = self.store.list("meta_traces", run_id=run_id)
        artifacts = self.artifact_store.list_artifacts(run_id=run_id)
        if not traces:
            raise ValueError(f"No traces found for run_id={run_id}")
        workspace_id = traces[0]["workspace_id"]
        objective = self._objective(traces)
        next_version = self._next_version(run_id)
        memory = RunMemory(
            run_id=run_id,
            version=next_version,
            workspace_id=workspace_id,
            objective=objective,
            current_status=self._status(traces),
            best_metrics=self._collect_metrics(artifacts),
            key_decisions=self._key_decisions(traces),
            blockers=self._blockers(traces),
            next_actions=self._next_actions(traces),
            source_trace_ids=[trace["trace_id"] for trace in traces],
            metadata=self._metadata(traces, artifacts),
        )
        self.store.upsert(
            "run_memories",
            f"{run_id}:v{memory.version}",
            memory,
            workspace_id=workspace_id,
            run_id=run_id,
        )
        return memory

    def _objective(self, traces: list[dict[str, Any]]) -> str:
        for trace in traces:
            objective = trace.get("metadata", {}).get("objective")
            if objective:
                return str(objective)
        return str(traces[0]["task"])

    def _next_version(self, run_id: str) -> int:
        existing = self.store.list("run_memories", run_id=run_id)
        versions = [int(item.get("version", 0)) for item in existing]
        return max(versions, default=0) + 1

    def _status(self, traces: list[dict[str, Any]]) -> str:
        if any(trace["status"] == "failed" for trace in traces):
            return "failed"
        if all(trace["status"] in {"succeeded", "skipped"} for trace in traces):
            return "succeeded"
        return "running"

    def _key_decisions(self, traces: list[dict[str, Any]]) -> list[str]:
        decisions: list[str] = []
        for trace in traces:
            for finding in trace.get("findings", []):
                if any(keyword in finding.lower() for keyword in DECISION_KEYWORDS):
                    decisions.append(finding)
        return decisions

    def _blockers(self, traces: list[dict[str, Any]]) -> list[str]:
        blockers: list[str] = []
        for trace in traces:
            blockers.extend(trace.get("limitations", []))
            if trace["status"] == "failed":
                blockers.append(f"failed trace: {trace['task']}")
        return blockers

    def _next_actions(self, traces: list[dict[str, Any]]) -> list[str]:
        actions = [trace.get("next_action") for trace in traces if trace.get("next_action")]
        return list(dict.fromkeys(actions[-5:]))

    def _collect_metrics(self, artifacts: list[Any]) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        for artifact in artifacts:
            for key, value in artifact.metrics.items():
                if isinstance(value, (int, float)) or key in SCALAR_METRIC_KEYS:
                    metrics[key] = value
            if artifact.uri.endswith("json") and artifact.metadata.get("filename") == "optical_metrics.json":
                path = self.artifact_store.resolve_uri(artifact.uri)
                if path.exists():
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    for key, value in payload.items():
                        if isinstance(value, (int, float)) or key in SCALAR_METRIC_KEYS:
                            metrics[key] = value
        return metrics

    def _metadata(self, traces: list[dict[str, Any]], artifacts: list[Any]) -> dict[str, Any]:
        backend = "mock_deeplens"
        for trace in reversed(traces):
            candidate = trace.get("metadata", {}).get("backend")
            if candidate:
                backend = str(candidate)
                break
        extra: dict[str, Any] = {}
        for artifact in artifacts:
            if artifact.metadata.get("backend") == backend:
                extra = {
                    "backend_capability_level": artifact.metadata.get("backend_capability_level"),
                    "encoder_behavior_realized": artifact.metadata.get("encoder_behavior_realized"),
                    "encoder_behavior_realization_level": artifact.metadata.get("encoder_behavior_realization_level"),
                    "physical_validation_level": artifact.metadata.get("physical_validation_level"),
                    "proxy_transform_applied": artifact.metadata.get("proxy_transform_applied"),
                    "proxy_transform_name": artifact.metadata.get("proxy_transform_name"),
                    "selected_realization_level": artifact.metadata.get("selected_realization_level"),
                    "semi_native_attempted": artifact.metadata.get("semi_native_attempted"),
                    "semi_native_succeeded": artifact.metadata.get("semi_native_succeeded"),
                    "proxy_fallback_used": artifact.metadata.get("proxy_fallback_used"),
                    "claim_scope": artifact.metadata.get("claim_scope"),
                    "deeplens_version": artifact.metadata.get("deeplens_version"),
                    "python_executable": artifact.metadata.get("python_executable"),
                }
                break
        return backend_metadata(backend, {key: value for key, value in extra.items() if value is not None})
