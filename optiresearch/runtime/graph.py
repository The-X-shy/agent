"""MVP runtime flow."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from optiresearch.agents.critical_reviewer import CriticalReviewer
from optiresearch.agents.lead_investigator import LeadInvestigator
from optiresearch.agents.method_builder import MethodBuilder
from optiresearch.agents.simulation_experimentalist import SimulationExperimentalist
from optiresearch.memory.compiler import MemoryCompiler
from optiresearch.memory.meta_trace import MetaTraceWriter
from optiresearch.memory.router import MemoryRouter
from optiresearch.memory.schemas import MetaTrace, make_trace_id
from optiresearch.memory.plan_template import PlanTemplateManager
from optiresearch.memory.skill_memory import SkillMemoryManager
from optiresearch.runtime.backend_metadata import backend_metadata
from optiresearch.schemas.experiment import ExperimentSpec
from optiresearch.skills.executor import SkillExecutor
from optiresearch.storage.file_artifact_store import FileArtifactStore
from optiresearch.storage.sqlite_store import SQLiteStore


def run_mvp_flow(
    objective: str,
    workspace_id: str = "default",
    experiment_spec: Optional[ExperimentSpec] = None,
    backend: Optional[str] = None,
    use_llm: bool = False,
    llm_provider: Any = None,
    realization: str = "auto",
) -> dict[str, Any]:
    """Run the full rule-based MVP research loop."""

    store = SQLiteStore()
    store.init_db()
    artifact_store = FileArtifactStore(store=store)
    trace_writer = MetaTraceWriter(store)
    lead = LeadInvestigator()
    method_builder = MethodBuilder()
    experimentalist = SimulationExperimentalist()
    executor = SkillExecutor()

    selected_backend = backend or (experiment_spec.backend if experiment_spec else "mock_deeplens")
    if selected_backend not in {"mock_deeplens", "deeplens"}:
        raise ValueError(f"Unsupported backend: {selected_backend}")
    plan = lead.plan(objective, workspace_id=workspace_id, backend=selected_backend, use_llm=use_llm, provider=llm_provider)
    plan["first_run"]["realization"] = realization
    lead_trace = trace_writer.write_trace(_lead_trace(plan))
    method = experiment_spec or (
        method_builder.build_experiment_spec_with_llm(objective, {}, provider=llm_provider)
        if use_llm
        else method_builder.build_mock_optical_spec(objective, backend=selected_backend)
    )
    if method.backend != selected_backend:
        method = method_builder.build_mock_optical_spec(objective, encoder_type=method.optical_spec.encoder_type, backend=selected_backend)
    method_trace = trace_writer.write_trace(_method_trace(plan, method, parent=lead_trace.trace_id))
    output_dir = _raw_output_dir(plan["run_id"], selected_backend)
    sim = experimentalist.run_simulation(
        plan=plan,
        method=method,
        skill_runtime=executor,
        artifact_store=artifact_store,
        trace_writer=trace_writer,
        output_dir=output_dir,
    )
    compiler = MemoryCompiler(store=store, artifact_store=artifact_store)
    run_memory = compiler.compile_run_memory(plan["run_id"])
    PlanTemplateManager(store).compile_from_run(plan["run_id"])
    SkillMemoryManager(store).update_from_run(plan["run_id"])
    reviewer = CriticalReviewer(store=store, artifact_store=artifact_store, workspace_id=workspace_id)
    claims = reviewer.review_claims(plan["run_id"], trace_writer=trace_writer)
    router = MemoryRouter(store=store, artifact_store=artifact_store)
    context_pack = router.query(
        role="CriticalReviewer",
        intent="evidence claim",
        query=objective,
        scope={"run_id": plan["run_id"], "workspace_id": workspace_id},
        require_evidence=True,
    )
    traces = trace_writer.list_traces(run_id=plan["run_id"])
    artifacts = artifact_store.list_artifacts(run_id=plan["run_id"])
    return {
        "run_id": plan["run_id"],
        "trace_ids": [trace.trace_id for trace in traces],
        "artifact_ids": [artifact.artifact_id for artifact in artifacts],
        "experiment_spec": method.model_dump(mode="json"),
        "run_memory": run_memory.model_dump(mode="json"),
        "claims": [claim.model_dump(mode="json") for claim in claims],
        "context_pack": context_pack,
        "errors": sim["errors"],
    }


def _raw_output_dir(run_id: str, backend: str = "mock_deeplens") -> Path:
    artifact_root = Path(os.getenv("OPTIRESEARCH_ARTIFACT_ROOT", "./workspace/artifacts"))
    name = "deeplens_psf" if backend == "deeplens" else "mock_psf"
    return artifact_root.parent / "runs" / run_id / name


def _lead_trace(plan: dict[str, Any]) -> MetaTrace:
    backend = plan.get("first_run", {}).get("backend", "mock_deeplens")
    task = f"plan first {backend} optical research run"
    now = datetime.now(timezone.utc)
    trace = MetaTrace(
        trace_id=make_trace_id(plan["workspace_id"], plan["run_id"], "lead-plan", "LeadInvestigator", task),
        workspace_id=plan["workspace_id"],
        run_id=plan["run_id"],
        branch_id=None,
        step_id="lead-plan",
        actor="LeadInvestigator",
        phase="Explore",
        task=task,
        skill_id=None,
        skill_version=None,
        tool=None,
        input_refs=[],
        output_refs=[],
        findings=[f"decision: start with {backend} and evidence review skills"],
        limitations=[] if backend == "deeplens" else ["real DeepLens backend is not required for mock runs"],
        next_action=f"build {backend} optical spec",
        status="succeeded",
        timestamp_start=now,
        timestamp_end=now,
        parents=[],
        content_hash=None,
        metadata={"objective": plan["objective"], "candidate_skills": plan["candidate_skills"], **backend_metadata(backend)},
    )
    if plan.get("llm_metadata"):
        trace.metadata.update(plan["llm_metadata"])
    return trace


def _method_trace(plan: dict[str, Any], method: ExperimentSpec, parent: str) -> MetaTrace:
    task = f"build {method.backend} optical simulation spec"
    now = datetime.now(timezone.utc)
    return MetaTrace(
        trace_id=make_trace_id(plan["workspace_id"], plan["run_id"], "method", "MethodBuilder", task),
        workspace_id=plan["workspace_id"],
        run_id=plan["run_id"],
        branch_id=None,
        step_id="method",
        actor="MethodBuilder",
        phase="Explore",
        task=task,
        skill_id="deeplens-adapter",
        skill_version="0.1.0",
        tool=None,
        input_refs=[],
        output_refs=[],
        findings=[
            f"decision: use 9 depth planes and 31 wavelength bands for the {method.backend} sweep",
            f"metric targets: {method.metric_spec.thresholds}",
        ],
        limitations=[],
        next_action=f"run {method.backend} PSF simulation",
        status="succeeded",
        timestamp_start=now,
        timestamp_end=now,
        parents=[parent],
        content_hash=None,
        metadata={
            "objective": plan["objective"],
            "experiment_spec": method.model_dump(mode="json"),
            **backend_metadata(method.backend),
        },
    )
