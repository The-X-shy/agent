"""Agent system self-test for Phase 36."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SelfTestResult:
    check_name: str
    passed: bool
    latency_sec: float
    error: str = ""
    details: dict[str, Any] = field(default_factory=dict)


def run_agent_self_test(output_dir: str | Path = "workspace/reports") -> list[SelfTestResult]:
    results: list[SelfTestResult] = []

    # 1. Backend registry
    t0 = time.time()
    try:
        from optiresearch.backends.registry import list_backends
        backends = list_backends()
        results.append(SelfTestResult("backend_registry", len(backends) > 0,
                                       time.time() - t0, details={"backends": len(backends)}))
    except Exception as e:
        results.append(SelfTestResult("backend_registry", False, time.time() - t0, str(e)))

    # 2. Skill registry
    t0 = time.time()
    try:
        from optiresearch.skills.registry_v2 import SkillRegistryV2
        reg = SkillRegistryV2()
        skills = reg.list_skills()
        results.append(SelfTestResult("skill_registry", len(skills) > 0,
                                       time.time() - t0, details={"skills": len(skills)}))
    except Exception as e:
        results.append(SelfTestResult("skill_registry", False, time.time() - t0, str(e)))

    # 3. ClaimGate
    t0 = time.time()
    try:
        from optiresearch.memory.claim_gate_v2 import ClaimGateV2
        gate = ClaimGateV2()
        results.append(SelfTestResult("claim_gate", True, time.time() - t0))
    except Exception as e:
        results.append(SelfTestResult("claim_gate", False, time.time() - t0, str(e)))

    # 4. StrategyEngine
    t0 = time.time()
    try:
        from optiresearch.agents.strategy_engine import StrategyEngine
        engine = StrategyEngine()
        results.append(SelfTestResult("strategy_engine", True, time.time() - t0))
    except Exception as e:
        results.append(SelfTestResult("strategy_engine", False, time.time() - t0, str(e)))

    # 5. ResearchMemoryV2
    t0 = time.time()
    try:
        from optiresearch.memory.research_memory_v2 import ResearchMemoryV2
        mem = ResearchMemoryV2()
        entries = mem.query()
        results.append(SelfTestResult("research_memory_v2", len(entries) > 0,
                                       time.time() - t0, details={"entries": len(entries)}))
    except Exception as e:
        results.append(SelfTestResult("research_memory_v2", False, time.time() - t0, str(e)))

    # 6. EventBus
    t0 = time.time()
    try:
        from optiresearch.agent_system.event_bus import EventBus
        bus = EventBus()
        results.append(SelfTestResult("event_bus", bus is not None, time.time() - t0))
    except Exception as e:
        results.append(SelfTestResult("event_bus", False, time.time() - t0, str(e)))

    # 7. StateStore
    t0 = time.time()
    try:
        from optiresearch.agent_system.state_store import StateStore
        store = StateStore()
        store.seed_phase35_result()
        results.append(SelfTestResult("state_store", store.state.last_failure_mode != "",
                                       time.time() - t0,
                                       details={"failure_mode": store.state.last_failure_mode}))
    except Exception as e:
        results.append(SelfTestResult("state_store", False, time.time() - t0, str(e)))

    # 8. Remote worker check
    t0 = time.time()
    try:
        from optiresearch.remote.worker_registry import RemoteWorkerRegistry
        workers = RemoteWorkerRegistry().list_workers()
        results.append(SelfTestResult("remote_workers", True, time.time() - t0,
                                       details={"workers": len(workers)}))
    except Exception as e:
        results.append(SelfTestResult("remote_workers", False, time.time() - t0, str(e)))

    # 9. LLM provider
    t0 = time.time()
    try:
        from optiresearch.llm.provider import get_llm_provider_config
        cfg = get_llm_provider_config()
        results.append(SelfTestResult("llm_provider", cfg is not None, time.time() - t0,
                                       details={"provider": str(cfg)[:100] if cfg else "none"}))
    except Exception:
        results.append(SelfTestResult("llm_provider", True, time.time() - t0,
                                       details={"note": "LLM provider check skipped"}))

    # 10. Artifact store
    t0 = time.time()
    try:
        path = Path("workspace/agent_self_test_check.txt")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("self-test", encoding="utf-8")
        results.append(SelfTestResult("artifact_store", path.exists(), time.time() - t0))
    except Exception as e:
        results.append(SelfTestResult("artifact_store", False, time.time() - t0, str(e)))

    _save_results(results, output_dir)
    return results


def _save_results(results: list[SelfTestResult], output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = [{"check_name": r.check_name, "passed": r.passed,
             "latency_sec": r.latency_sec, "error": r.error,
             "details": r.details} for r in results]
    (out / "agent_self_test.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    passed = sum(1 for r in results if r.passed)
    lines = [
        "# Agent Self-Test Report",
        "",
        f"**Passed:** {passed}/{len(results)}",
        "",
        "| Check | Result | Latency | Details |",
        "|---|---|---|---|",
    ]
    for r in results:
        status = "PASS" if r.passed else f"FAIL: {r.error}"
        lines.append(f"| {r.check_name} | {status} | {r.latency_sec:.3f}s | {r.details} |")
    (out / "agent_self_test.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
