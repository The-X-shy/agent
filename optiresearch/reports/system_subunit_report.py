"""System subunit report for Phase 36."""

from __future__ import annotations

from pathlib import Path


def export_system_subunit_report(output_dir: str | Path = "workspace/reports") -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    lines = _build_report()
    path = out / "system_subunit_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _build_report() -> list[str]:
    return [
        "# Agent System Subunit Report",
        "",
        "## 1. Overview",
        "",
        "This report documents the complete agent system subunit architecture "
        "established in Phase 36. The system now has 9 interconnected subsystems "
        "that enable autonomous research iteration: from evidence ingestion through "
        "failure classification, strategy reasoning, experiment design, plan evaluation, "
        "skill execution, and reporting.",
        "",
        "## 2. Event Bus",
        "",
        "- **Module:** `optiresearch/agent_system/event_bus.py`",
        "- **Model:** In-memory pub/sub with JSON serialization",
        "- **Event types:** 17 (experiment_*, claim_*, strategy_*, skill_*, etc.)",
        "- **Subscribers:** Wildcard + type-specific handlers",
        "- **Export:** JSON via `export_events()`",
        "",
        "## 3. State Store",
        "",
        "- **Module:** `optiresearch/agent_system/state_store.py`",
        "- **Persistence:** JSON file-based at `workspace/agent_state/`",
        "- **Features:** load/save, update_from_event, snapshot, diff_snapshots",
        "- **Seeded with:** Phase 35 negative result (unstable_native_geolens_update)",
        "",
        "## 4. Skill Runtime v2",
        "",
        "- **Modules:** `skills/contracts.py`, `skills/registry_v2.py`, `skills/runtime_v2.py`",
        "- **Registered skills:** 9 (geolens_hsi, stabilization_sweep, backend_probe, claim_check, autograd_audit, strategy_recommendation, report_generation, remote_execution, evidence_registry_export)",
        "- **Features:** input validation, skill dispatch, audit, event publishing",
        "",
        "## 5. Failure Taxonomy",
        "",
        "- **Module:** `optiresearch/agent_system/failure_taxonomy.py`",
        "- **Built-in modes:** 8 (unstable_native_geolens_update, backend_unavailable, gradient_instability, claim_overreach, rollback_all_updates, remote_execution_failure, platform_incompatibility, metric_no_improvement)",
        "- **Classification:** Pattern-matching on evidence dicts",
        "- **Phase 35 failure:** classified as unstable_native_geolens_update (gradient_instability + rollback_all_updates)",
        "",
        "## 6. Recovery Policy",
        "",
        "- **Module:** `optiresearch/agent_system/recovery_policy.py`",
        "- **Recovery types:** 11 (try_alternative_parameterization, redesign_objective, switch_backend, probe_waveoptics_path, request_real_data, report_negative_result, etc.)",
        "- **Features:** ranked recoveries, strategy conversion, explanation",
        "",
        "## 7. Evidence-to-Strategy Reasoner",
        "",
        "- **Module:** `optiresearch/agents/evidence_strategy_reasoner.py`",
        "- **Generated strategies from Phase 35:** 6",
        "  1. alt_param_diffractive (alternative_parameterization)",
        "  2. objective_redesign_simpler_metric (objective_redesign)",
        "  3. backend_switch_waveoptics (waveoptics_probe)",
        "  4. report_negative_result (report_negative_result)",
        "  5. real_data_request (real_data_request)",
        "  6. param_reduction (parameter_reduction)",
        "- **Top strategy:** report_negative_result (lowest risk, highest immediate value)",
        "",
        "## 8. Experiment Design Generator",
        "",
        "- **Module:** `optiresearch/agents/experiment_design_generator.py`",
        "- **Output:** ExperimentDesignCandidate with concrete spec_payload",
        "- **6 designs generated from Phase 35 negative result**",
        "",
        "## 9. Candidate Plan Evaluator",
        "",
        "- **Module:** `optiresearch/agents/candidate_plan_evaluator.py`",
        "- **Scoring dimensions:** 8 (evidence_gain, metric_gain_likelihood, execution_feasibility, backend_availability, runtime_cost, claim_safety, novelty, risk_penalty)",
        "- **Recommendations:** execute_now, dry_run_first, needs_remote, needs_user_data, defer, reject",
        "",
        "## 10. Self-Test",
        "",
        "- **Module:** `optiresearch/agent_system/self_test.py`",
        "- **Checks:** 10 (backend_registry, skill_registry, claim_gate, strategy_engine, research_memory_v2, event_bus, state_store, remote_workers, llm_provider, artifact_store)",
        "",
        "## 11. Subunit Benchmark",
        "",
        "- **Module:** `optiresearch/benchmarks/agent_subunit_bench.py`",
        "- **Tasks:** 10 (claim_overreach_detection, backend_switch_recommendation, negative_result_classification, recovery_recommendation, experiment_design_generation, skill_selection, event_logging, state_update, plan_scoring, report_generation)",
        "",
        "## 12. How Phase 35 Negative Result is Converted into Candidate Plans",
        "",
        "1. **FailureClassifier** matches Phase 35 sweep results → `unstable_native_geolens_update`",
        "2. **RecoveryPolicy** generates 7 ranked recoveries from this failure mode",
        "3. **EvidenceStrategyReasoner** converts recoveries + context → 6 CandidateStrategies",
        "4. **ExperimentDesignGenerator** converts each strategy → ExperimentDesignCandidate",
        "5. **CandidatePlanEvaluator** scores and ranks designs across 8 dimensions",
        "6. **StateStore** records `last_failure_mode` and `pending_actions`",
        "7. **EventBus** publishes `negative_result_recorded`, `recovery_recommended` events",
        "8. The system is now ready for the next autonomous iteration",
        "",
        "## 13. Current Limitations",
        "",
        "- No automatic execution loop — designs are proposed but not auto-executed",
        "- LLMPlanner integration not yet wired to EvidenceStrategyReasoner",
        "- SkillRuntimeV2 only dispatches claim_check and strategy_recommendation; other skills need runtime implementations",
        "- Remote executor not integrated with EventBus",
        "- StateStore does not auto-snapshot on every event",
        "- Benchmark uses synthetic test data, not real experiment results",
    ]


def export_system_subunit_report_cli(output_dir: str = "workspace/reports") -> str:
    path = export_system_subunit_report(output_dir)
    return str(path)
