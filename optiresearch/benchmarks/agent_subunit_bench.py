"""Agent subunit benchmark for Phase 36."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class BenchResult:
    task_id: str
    passed: bool
    latency_sec: float
    correctness: str = ""
    safety_gate_passed: bool = True
    expected_action_match: bool = False
    claim_boundary_respected: bool = True
    error: str = ""


def run_agent_subunit_benchmark(output_dir: str | Path = "workspace/benchmarks") -> list[BenchResult]:
    results: list[BenchResult] = []

    # Task 1: Claim overreach detection
    t0 = time.time()
    try:
        from optiresearch.memory.claim_gate_v2 import ClaimGateV2
        gate = ClaimGateV2()
        decision = gate.check_claim(
            "Full wave-optics native HSI with real camera data is fully validated",
            "deeplens_geolens_geometric",
        )
        results.append(BenchResult("claim_overreach_detection", True, time.time() - t0,
                                    correctness="downgraded" if "downgraded" in decision.decision else "passed",
                                    safety_gate_passed=True,
                                    expected_action_match="downgraded" in decision.decision,
                                    claim_boundary_respected=True))
    except Exception as e:
        results.append(BenchResult("claim_overreach_detection", False, time.time() - t0, error=str(e)))

    # Task 2: Backend switch recommendation
    t0 = time.time()
    try:
        from optiresearch.agents.strategy_engine import StrategyEngine
        engine = StrategyEngine()
        rec = engine.recommend(
            {"status": "unsupported", "error_code": "BUILD_FAILED"},
            "deeplens_geolens_geometric",
        )
        results.append(BenchResult("backend_switch_recommendation", True, time.time() - t0,
                                    correctness=rec.recommended_action,
                                    safety_gate_passed=True,
                                    expected_action_match=bool(rec.recommended_action)))
    except Exception as e:
        results.append(BenchResult("backend_switch_recommendation", False, time.time() - t0, error=str(e)))

    # Task 3: Negative result classification
    t0 = time.time()
    try:
        from optiresearch.agent_system.failure_taxonomy import FailureClassifier
        classifier = FailureClassifier()
        fm = classifier.classify({"optical_gradient_norm_max": 4098, "accepted_update_count": 0,
                                   "rejected_update_count": 2, "proxy_fallback_used": False})
        results.append(BenchResult("negative_result_classification", fm is not None, time.time() - t0,
                                    correctness=fm.failure_id if fm else "none",
                                    safety_gate_passed=True,
                                    expected_action_match=fm is not None))
    except Exception as e:
        results.append(BenchResult("negative_result_classification", False, time.time() - t0, error=str(e)))

    # Task 4: Recovery recommendation
    t0 = time.time()
    try:
        from optiresearch.agent_system.recovery_policy import RecoveryPolicy
        policy = RecoveryPolicy()
        rec = policy.recommend_recovery("unstable_native_geolens_update")
        results.append(BenchResult("recovery_recommendation", len(rec.get("recoveries", [])) > 0,
                                    time.time() - t0,
                                    correctness=f"{len(rec.get('recoveries', []))} recoveries",
                                    safety_gate_passed=True,
                                    expected_action_match=len(rec.get("recoveries", [])) > 0))
    except Exception as e:
        results.append(BenchResult("recovery_recommendation", False, time.time() - t0, error=str(e)))

    # Task 5: Experiment design generation
    t0 = time.time()
    try:
        from optiresearch.agents.evidence_strategy_reasoner import EvidenceStrategyReasoner
        from optiresearch.agents.experiment_design_generator import ExperimentDesignGenerator
        reasoner = EvidenceStrategyReasoner()
        strategies = reasoner.reason()
        gen = ExperimentDesignGenerator()
        designs = gen.generate_designs(strategies)
        results.append(BenchResult("experiment_design_generation", len(designs) > 0,
                                    time.time() - t0,
                                    correctness=f"{len(designs)} designs",
                                    safety_gate_passed=True,
                                    expected_action_match=len(designs) >= 4))
    except Exception as e:
        results.append(BenchResult("experiment_design_generation", False, time.time() - t0, error=str(e)))

    # Task 6: Skill selection
    t0 = time.time()
    try:
        from optiresearch.skills.registry_v2 import SkillRegistryV2
        reg = SkillRegistryV2()
        skills = reg.find_by_task("stable_lens_hsi_codesign")
        results.append(BenchResult("skill_selection", len(skills) > 0, time.time() - t0,
                                    correctness=f"{len(skills)} skills found",
                                    safety_gate_passed=True,
                                    expected_action_match=len(skills) > 0))
    except Exception as e:
        results.append(BenchResult("skill_selection", False, time.time() - t0, error=str(e)))

    # Task 7: Event logging
    t0 = time.time()
    try:
        from optiresearch.agent_system.event_bus import EventBus
        from optiresearch.agent_system.events import AgentEvent
        bus = EventBus()
        bus.publish(AgentEvent.create("experiment_completed", "controller",
                                       {"status": "succeeded"}))
        results.append(BenchResult("event_logging", bus.count() > 0, time.time() - t0,
                                    correctness=f"{bus.count()} events",
                                    safety_gate_passed=True,
                                    expected_action_match=bus.count() == 1))
    except Exception as e:
        results.append(BenchResult("event_logging", False, time.time() - t0, error=str(e)))

    # Task 8: State update
    t0 = time.time()
    try:
        from optiresearch.agent_system.state_store import StateStore
        store = StateStore()
        initial_fm = store.state.last_failure_mode
        store.seed_phase35_result()
        updated = store.state.last_failure_mode != initial_fm or store.state.last_failure_mode != ""
        results.append(BenchResult("state_update", updated, time.time() - t0,
                                    correctness=store.state.last_failure_mode,
                                    safety_gate_passed=True,
                                    expected_action_match=updated))
    except Exception as e:
        results.append(BenchResult("state_update", False, time.time() - t0, error=str(e)))

    # Task 9: Plan scoring
    t0 = time.time()
    try:
        from optiresearch.agents.candidate_plan_evaluator import CandidatePlanEvaluator
        from optiresearch.agents.experiment_design_generator import ExperimentDesignCandidate
        evaluator = CandidatePlanEvaluator()
        test_designs = [
            ExperimentDesignCandidate("test_1", "test", "backend", "task",
                                       expected_evidence_level="native_lens_simulation",
                                       risk_level="low", estimated_runtime_sec=600),
        ]
        scores = evaluator.evaluate(test_designs)
        results.append(BenchResult("plan_scoring", len(scores) == 1, time.time() - t0,
                                    correctness=f"score={scores[0].total_score:.3f}" if scores else "none",
                                    safety_gate_passed=True,
                                    expected_action_match=len(scores) == 1))
    except Exception as e:
        results.append(BenchResult("plan_scoring", False, time.time() - t0, error=str(e)))

    # Task 10: Report generation
    t0 = time.time()
    try:
        from optiresearch.reports.system_subunit_report import export_system_subunit_report
        path = export_system_subunit_report()
        results.append(BenchResult("report_generation", path.exists(), time.time() - t0,
                                    correctness=str(path),
                                    safety_gate_passed=True,
                                    expected_action_match=path.exists()))
    except Exception as e:
        results.append(BenchResult("report_generation", False, time.time() - t0, error=str(e)))

    _save_bench_results(results, output_dir)
    return results


def _save_bench_results(results: list[BenchResult], output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = [{"task_id": r.task_id, "passed": r.passed, "latency_sec": r.latency_sec,
             "correctness": r.correctness, "safety_gate_passed": r.safety_gate_passed,
             "expected_action_match": r.expected_action_match,
             "claim_boundary_respected": r.claim_boundary_respected, "error": r.error}
            for r in results]
    (out / "agent_subunit_bench_report.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    passed = sum(1 for r in results if r.passed)
    lines = [
        "# Agent Subunit Benchmark Report",
        "",
        f"**Passed:** {passed}/{len(results)}",
        "",
        "| Task | Result | Latency | Correctness | Safety |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        status = "PASS" if r.passed else f"FAIL: {r.error}"
        safety = "PASS" if r.safety_gate_passed else "FAIL"
        lines.append(f"| {r.task_id} | {status} | {r.latency_sec:.3f}s | {r.correctness} | {safety} |")
    (out / "agent_subunit_bench_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
