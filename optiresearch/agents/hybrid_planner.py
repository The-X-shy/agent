"""Hybrid Planner for Phase 37 — merges rule-based EvidenceStrategyReasoner with LLMPlanner."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

from optiresearch.agents.evidence_strategy_reasoner import CandidateStrategy, EvidenceStrategyReasoner
from optiresearch.agents.experiment_design_generator import ExperimentDesignCandidate, ExperimentDesignGenerator
from optiresearch.agents.candidate_plan_evaluator import CandidatePlanEvaluator, PlanScore


PlannerMode = Literal["rule_only", "llm_only", "llm_with_rule_context", "llm_with_rule_fallback"]


@dataclass
class HybridPlanResult:
    mode: str
    strategies_count: int = 0
    designs_count: int = 0
    top_design_id: str = ""
    top_score: float = 0.0
    top_recommendation: str = ""
    llm_called: bool = False
    llm_proposals_count: int = 0
    merged_proposals_count: int = 0
    errors: list[str] = field(default_factory=list)
    strategies: list[dict[str, Any]] = field(default_factory=list)
    designs: list[dict[str, Any]] = field(default_factory=list)
    scores: list[dict[str, Any]] = field(default_factory=list)


class HybridPlanner:
    def __init__(self):
        self._reasoner = EvidenceStrategyReasoner()
        self._gen = ExperimentDesignGenerator()
        self._evaluator = CandidatePlanEvaluator()

    def plan(
        self,
        objective: str = "recover from native GeoLens instability",
        mode: PlannerMode = "rule_only",
        llm_provider: str = "",
        backend_id: str = "deeplens_geolens_geometric",
        failure_mode: str = "unstable_native_geolens_update",
        max_strategies: int = 6,
    ) -> HybridPlanResult:
        result = HybridPlanResult(mode=mode)
        strategies: list[CandidateStrategy] = []

        # Rule-based strategies
        rule_strategies = self._reasoner.reason(
            objective=objective, failure_mode=failure_mode, backend_id=backend_id,
        )

        if mode == "rule_only":
            strategies = rule_strategies[:max_strategies]
        elif mode == "llm_only":
            if llm_provider:
                strategies = self._call_llm(objective, failure_mode, llm_provider, [])
                result.llm_called = True
                result.llm_proposals_count = len(strategies)
            else:
                result.errors.append("LLM provider required for llm_only mode; falling back to rule")
                strategies = rule_strategies[:max_strategies]
        elif mode == "llm_with_rule_context":
            strategies = rule_strategies[:max_strategies]
            if llm_provider:
                llm_strategies = self._call_llm(objective, failure_mode, llm_provider, rule_strategies)
                result.llm_called = True
                result.llm_proposals_count = len(llm_strategies)
                strategies = self._merge(rule_strategies[:max_strategies], llm_strategies, max_strategies)
        elif mode == "llm_with_rule_fallback":
            if llm_provider:
                strategies = self._call_llm(objective, failure_mode, llm_provider, [])
                result.llm_called = True
                result.llm_proposals_count = len(strategies)
            if not strategies:
                strategies = rule_strategies[:max_strategies]
                result.errors.append("LLM produced no proposals; using rule-based fallback")

        result.strategies_count = len(strategies)
        result.strategies = [{"strategy_id": s.strategy_id, "strategy_type": s.strategy_type,
                              "rationale": s.rationale[:150]} for s in strategies]

        # Generate designs
        designs = self._gen.generate_designs(strategies)
        result.designs_count = len(designs)
        result.designs = [{"design_id": d.design_id, "backend_id": d.backend_id,
                           "task_type": d.task_type, "risk_level": d.risk_level} for d in designs]

        # Score
        scores = self._evaluator.evaluate(designs)
        result.merged_proposals_count = len(scores)
        result.scores = [{"design_id": s.design_id, "total_score": s.total_score,
                          "recommendation": s.recommendation} for s in scores[:5]]

        if scores:
            result.top_design_id = scores[0].design_id
            result.top_score = scores[0].total_score
            result.top_recommendation = scores[0].recommendation

        return result

    def _call_llm(self, objective: str, failure_mode: str, provider: str,
                  rule_strategies: list[CandidateStrategy]) -> list[CandidateStrategy]:
        try:
            context = self._build_reasoner_context(objective, failure_mode, rule_strategies)
            from optiresearch.agents.llm_planner import LLMPlanner
            planner = LLMPlanner()
            proposals = planner.plan(
                objective=objective,
                context=context,
                provider=provider,
            )
            strategies = []
            for p in proposals[:6]:
                strategies.append(CandidateStrategy(
                    strategy_id=p.proposal_id,
                    strategy_type="llm_proposal",
                    rationale=p.rationale or "",
                    expected_evidence_gain="unknown",
                    expected_metric_gain="unknown",
                    risk="medium",
                    cost="medium",
                ))
            return strategies
        except Exception:
            return []

    def _build_reasoner_context(self, objective: str, failure_mode: str,
                                rule_strategies: list[CandidateStrategy]) -> dict[str, Any]:
        return {
            "objective": objective,
            "failure_mode": failure_mode,
            "rule_based_strategies": [
                {"id": s.strategy_id, "type": s.strategy_type, "rationale": s.rationale[:200]}
                for s in rule_strategies
            ],
            "phase35_summary": (
                "Phase 35 tested 30 hyperparameter configurations with trust region, "
                "PSF stability gating, and accept tolerance. 0/30 configs achieved "
                "accepted optical updates. The GeoLensCooke geometric parameterization "
                "appears fundamentally unstable for gradient-based optimization."
            ),
        }

    def _merge(self, rule: list[CandidateStrategy], llm: list[CandidateStrategy],
               max_count: int) -> list[CandidateStrategy]:
        seen = {s.strategy_id for s in rule}
        merged = list(rule)
        for s in llm:
            if s.strategy_id not in seen and len(merged) < max_count:
                merged.append(s)
                seen.add(s.strategy_id)
        return merged

    def export(self, result: HybridPlanResult, output_path: str | Path | None = None) -> Path:
        path = Path(output_path or "workspace/reports/hybrid_plan_result.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "mode": result.mode, "strategies_count": result.strategies_count,
            "designs_count": result.designs_count, "top_design_id": result.top_design_id,
            "top_score": result.top_score, "top_recommendation": result.top_recommendation,
            "llm_called": result.llm_called, "merged_proposals_count": result.merged_proposals_count,
            "errors": result.errors,
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return path
