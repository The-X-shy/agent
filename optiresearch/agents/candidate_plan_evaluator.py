"""Candidate Plan Evaluator for Phase 36.

Scores and ranks ExperimentDesignCandidates across 8 dimensions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from optiresearch.agents.experiment_design_generator import ExperimentDesignCandidate


@dataclass
class PlanScore:
    design_id: str
    total_score: float
    score_breakdown: dict[str, float] = field(default_factory=dict)
    recommendation: str = ""
    reason: str = ""


class CandidatePlanEvaluator:
    def __init__(self):
        self._weights = {
            "evidence_gain_score": 0.20,
            "metric_gain_likelihood": 0.15,
            "execution_feasibility": 0.15,
            "backend_availability": 0.15,
            "runtime_cost": 0.10,
            "claim_safety": 0.10,
            "novelty": 0.05,
            "risk_penalty": 0.10,
        }

    def evaluate(self, designs: list[ExperimentDesignCandidate]) -> list[PlanScore]:
        scores = []
        for d in designs:
            breakdown = self._score_design(d)
            total = sum(breakdown.values())
            rec, reason = self._recommend(total, d)
            scores.append(PlanScore(
                design_id=d.design_id,
                total_score=round(total, 3),
                score_breakdown=breakdown,
                recommendation=rec,
                reason=reason,
            ))
        scores.sort(key=lambda s: s.total_score, reverse=True)
        return scores

    def _score_design(self, d: ExperimentDesignCandidate) -> dict[str, float]:
        evidence_map = {"negative_result": 0.3, "native_lens_simulation": 0.5,
                        "native_waveoptics_simulation": 0.8, "real_hsi": 1.0,
                        "sweep_analysis": 0.4, "": 0.1}
        risk_map = {"low": 1.0, "medium": 0.7, "high": 0.4}
        rt_map = {0: 1.0, 60: 1.0, 600: 0.8, 3600: 0.5, 7200: 0.3}
        return {
            "evidence_gain_score": evidence_map.get(d.expected_evidence_level, 0.3) * self._weights["evidence_gain_score"],
            "metric_gain_likelihood": (0.3 if "gradient_instability" in d.expected_failure_modes else 0.6) * self._weights["metric_gain_likelihood"],
            "execution_feasibility": risk_map.get(d.risk_level, 0.7) * self._weights["execution_feasibility"],
            "backend_availability": (0.8 if d.backend_id else 1.0) * self._weights["backend_availability"],
            "runtime_cost": rt_map.get(d.estimated_runtime_sec, 0.5) * self._weights["runtime_cost"],
            "claim_safety": (0.9 if d.claim_ceiling else 0.5) * self._weights["claim_safety"],
            "novelty": (0.8 if "waveoptics" in d.design_id or "real" in d.design_id else 0.4) * self._weights["novelty"],
            "risk_penalty": -risk_map.get(d.risk_level, 0.7) * self._weights["risk_penalty"],
        }

    def _recommend(self, total: float, d: ExperimentDesignCandidate) -> tuple[str, str]:
        if d.estimated_runtime_sec == 0:
            return "needs_user_data", "Requires external data — cannot execute autonomously"
        if total >= 0.6:
            return "execute_now", "High score with acceptable risk"
        if total >= 0.45:
            return "dry_run_first", "Medium score — validate preconditions before execution"
        if "remote" in d.design_id or d.estimated_runtime_sec > 1800:
            return "needs_remote", "Requires remote execution or long runtime"
        if total < 0.3:
            return "defer", "Low score — revisit after gathering more evidence"
        return "execute_now", "Acceptable score"

    def export(self, scores: list[PlanScore],
               output_path: str | Path | None = None) -> Path:
        path = Path(output_path or "workspace/reports/candidate_plan_scores.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [{"design_id": s.design_id, "total_score": s.total_score,
                 "score_breakdown": s.score_breakdown,
                 "recommendation": s.recommendation, "reason": s.reason}
                for s in scores]
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def export_markdown(self, scores: list[PlanScore],
                        output_path: str | Path | None = None) -> Path:
        path = Path(output_path or "workspace/reports/candidate_plan_scores.md")
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Candidate Plan Scores",
            "",
            "| Rank | Design | Score | Recommendation | Reason |",
            "|---|---|---|---|---|",
        ]
        for i, s in enumerate(scores, 1):
            lines.append(
                f"| {i} | {s.design_id} | {s.total_score:.3f} | "
                f"{s.recommendation} | {s.reason} |"
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path
