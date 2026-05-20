"""Plan template memory manager."""

from __future__ import annotations

from typing import Any, Optional

from optiresearch.memory.schemas import PlanTemplate
from optiresearch.storage.sqlite_store import SQLiteStore


class PlanTemplateManager:
    """Create, match, and compile reusable plan templates."""

    def __init__(self, store: Optional[SQLiteStore] = None) -> None:
        self.store = store or SQLiteStore()
        self.store.init_db()

    def create_default_templates(self) -> list[PlanTemplate]:
        templates = [
            PlanTemplate(
                template_id="evaluate_mock_optical_encoder",
                intent="evaluate mock optical encoder",
                description="Run a deterministic mock optical simulation and review claims against metrics.",
                slots=["objective", "backend", "metrics"],
                steps=[
                    "build ExperimentSpec",
                    "run deeplens-adapter mock simulation",
                    "register artifacts",
                    "compile RunMemory",
                    "review claims",
                ],
                preconditions=["mock_deeplens backend available"],
                historical_success_rate=0.0,
                average_cost={"runs": 1},
                metadata={"used_skill_ids": ["deeplens-adapter", "evidence-review"]},
            ),
            PlanTemplate(
                template_id="evaluate_edof_hsi_encoder",
                intent="evaluate edof hsi encoder",
                description="Evaluate an EDOF-HSI encoder with optical and evidence metrics.",
                slots=["objective", "wavelength_bands", "depth_planes"],
                steps=["build ExperimentSpec", "simulate PSF", "evaluate metrics", "review evidence"],
                preconditions=["optical spec is available"],
                historical_success_rate=0.0,
                average_cost={"runs": 1},
                metadata={"used_skill_ids": ["deeplens-adapter"]},
            ),
            PlanTemplate(
                template_id="claim_evidence_review",
                intent="claim evidence review",
                description="Review claims against artifact metrics and source traces.",
                slots=["claim_text", "artifact_ids", "metric_names"],
                steps=["locate metrics artifacts", "attach EvidenceEdge", "explain claim"],
                preconditions=["claim exists"],
                historical_success_rate=0.0,
                average_cost={"reviews": 1},
                metadata={"used_skill_ids": ["evidence-review"]},
            ),
        ]
        for template in templates:
            existing = self.get(template.template_id)
            if existing is None:
                self.save(template)
        return self.list_templates()

    def save(self, template: PlanTemplate) -> PlanTemplate:
        self.store.upsert("plan_templates", template.template_id, template)
        return template

    def get(self, template_id: str) -> Optional[PlanTemplate]:
        payload = self.store.get("plan_templates", template_id)
        return PlanTemplate(**payload) if payload else None

    def list_templates(self) -> list[PlanTemplate]:
        return [PlanTemplate(**payload) for payload in self.store.list("plan_templates")]

    def match(self, intent: str, slots: Optional[dict[str, Any]] = None, top_k: int = 3) -> list[PlanTemplate]:
        self.create_default_templates()
        query_tokens = set(intent.lower().replace("-", " ").split())
        scored: list[tuple[float, PlanTemplate]] = []
        for template in self.list_templates():
            text = " ".join([template.template_id, template.intent, template.description, *template.slots]).lower()
            score = sum(1 for token in query_tokens if token in text) + template.historical_success_rate
            if slots:
                score += sum(0.2 for key in slots if key in template.slots)
            scored.append((score, template))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [template for score, template in scored[:top_k] if score > 0]

    def compile_from_run(self, run_id: str) -> PlanTemplate:
        traces = self.store.list("meta_traces", run_id=run_id)
        artifacts = self.store.list("artifacts", run_id=run_id)
        self.create_default_templates()
        template = self.get("evaluate_mock_optical_encoder")
        if template is None:
            raise ValueError("Default template was not created.")
        succeeded = traces and all(trace["status"] in {"succeeded", "skipped"} for trace in traces)
        skill_ids = sorted({trace.get("skill_id") for trace in traces if trace.get("skill_id")})
        artifact_types = sorted({artifact.get("metadata", {}).get("artifact_type", "unknown") for artifact in artifacts})
        metric_names = sorted({key for artifact in artifacts for key in artifact.get("metrics", {})})
        if succeeded:
            template.success_count += 1
        else:
            template.failure_count += 1
        total = template.success_count + template.failure_count
        template.historical_success_rate = round(template.success_count / total, 6) if total else 0.0
        template.average_cost = {
            "runs": total,
            "artifact_count": len(artifacts),
            "trace_count": len(traces),
        }
        template.metadata.update(
            {
                "source_run_ids": sorted(set([*template.metadata.get("source_run_ids", []), run_id])),
                "source_trace_ids": [trace["trace_id"] for trace in traces],
                "used_skill_ids": skill_ids,
                "artifact_types": artifact_types,
                "metric_names": metric_names,
            }
        )
        self.save(template)
        return template


class PlanTemplateMemory(PlanTemplateManager):
    """Backward-compatible alias for MVP code."""
