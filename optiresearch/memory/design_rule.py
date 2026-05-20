"""Design rule memory manager."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from optiresearch.memory.claim_evidence import ClaimEvidenceManager
from optiresearch.memory.schemas import DesignRule, make_deterministic_id
from optiresearch.storage.sqlite_store import SQLiteStore


class DesignRuleManager:
    """Compile and manage design rules from claims and artifact metrics."""

    def __init__(self, store: Optional[SQLiteStore] = None, workspace_id: str = "default") -> None:
        self.store = store or SQLiteStore()
        self.store.init_db()
        self.workspace_id = workspace_id

    def compile_from_claims(self) -> list[DesignRule]:
        comparison = self._encoder_metrics()
        controlled = comparison.get("controlled_chromatic_edof")
        achromatic = comparison.get("achromatic")
        rules: list[DesignRule] = []
        if controlled and achromatic:
            controlled_joint = self._joint(controlled)
            achromatic_joint = self._joint(achromatic)
            if controlled_joint > achromatic_joint:
                statement = (
                    "controlled chromatic EDOF gives better joint depth-spectral tradeoff "
                    "than fully achromatic mock encoder under current mock setting."
                )
                supported_by = sorted(
                    set(controlled.get("claim_ids", []) + achromatic.get("claim_ids", []) + [controlled["artifact_id"], achromatic["artifact_id"]])
                )
                rule = DesignRule(
                    rule_id=make_deterministic_id("rule", statement, controlled["run_id"], achromatic["run_id"]),
                    statement=statement,
                    scope=["mock_deeplens", "baseline_comparison", "edof_hsi"],
                    status="active",
                    confidence=round(min(0.95, 0.6 + controlled_joint - achromatic_joint), 6),
                    supported_by=supported_by,
                    contradicted_by=[],
                    valid_conditions={
                        "controlled_chromatic_edof": controlled,
                        "achromatic": achromatic,
                        "controlled_joint_score": controlled_joint,
                        "achromatic_joint_score": achromatic_joint,
                    },
                    invalid_at=None,
                    superseded_by=None,
                    source_trace_ids=sorted(
                        set(controlled.get("trace_ids", []) + achromatic.get("trace_ids", []))
                    ),
                )
                self.save(rule)
                rules.append(rule)
        return rules or self.list_rules()

    def detect_contradictions(self) -> list[dict[str, Any]]:
        metrics = self._encoder_metrics()
        achromatic = metrics.get("achromatic")
        controlled = metrics.get("controlled_chromatic_edof")
        if not achromatic or not controlled:
            return []
        contradictions: list[dict[str, Any]] = []
        manager = ClaimEvidenceManager(self.store, workspace_id=self.workspace_id)
        for payload in self.store.list("claims"):
            text = payload.get("text", "").lower()
            if "achromatic encoder is best for spectral separability" not in text:
                continue
            if controlled["metrics"].get("spectral_separability", 0.0) > achromatic["metrics"].get("spectral_separability", 0.0):
                claim = manager.get_claim(payload["claim_id"])
                if claim is None:
                    continue
                if not claim.support_edges:
                    manager.attach_support(
                        claim.claim_id,
                        achromatic["artifact_id"],
                        0.4,
                        relation="qualifies",
                    )
                manager.attach_contradiction(
                    claim.claim_id,
                    controlled["artifact_id"],
                    0.9,
                    relation="contradicts",
                )
                updated = manager.review_claim(claim.claim_id)
                contradictions.append(
                    {
                        "claim_id": updated.claim_id,
                        "status": updated.status,
                        "contradicting_artifact_id": controlled["artifact_id"],
                        "metric_name": "spectral_separability",
                        "controlled_value": controlled["metrics"]["spectral_separability"],
                        "achromatic_value": achromatic["metrics"]["spectral_separability"],
                    }
                )
        return contradictions

    def supersede_rule(self, rule_id: str, superseded_by: str) -> DesignRule:
        rule = self.get_rule(rule_id)
        if rule is None:
            raise KeyError(f"Unknown rule_id={rule_id}")
        rule.status = "superseded"
        rule.superseded_by = superseded_by
        rule.invalid_at = datetime.now(timezone.utc)
        self.save(rule)
        return rule

    def explain_rule(self, rule_id: str) -> dict[str, Any]:
        rule = self.get_rule(rule_id)
        if rule is None:
            raise KeyError(f"Unknown rule_id={rule_id}")
        evidence = []
        for ref in rule.supported_by:
            artifact = self.store.get("artifacts", ref)
            claim = self.store.get("claims", ref)
            if artifact:
                evidence.append(
                    {
                        "type": "artifact",
                        "id": ref,
                        "metrics": artifact.get("metrics", {}),
                        "uri": artifact.get("uri"),
                    }
                )
            elif claim:
                evidence.append({"type": "claim", "id": ref, "text": claim.get("text"), "status": claim.get("status")})
        return {
            "rule_id": rule.rule_id,
            "statement": rule.statement,
            "status": rule.status,
            "confidence": rule.confidence,
            "supported_by": rule.supported_by,
            "contradicted_by": rule.contradicted_by,
            "source_trace_ids": rule.source_trace_ids,
            "valid_conditions": rule.valid_conditions,
            "evidence": evidence,
        }

    def save(self, rule: DesignRule) -> DesignRule:
        self.store.upsert("design_rules", rule.rule_id, rule)
        return rule

    def get_rule(self, rule_id: str) -> Optional[DesignRule]:
        payload = self.store.get("design_rules", rule_id)
        return DesignRule(**payload) if payload else None

    def list_rules(self) -> list[DesignRule]:
        return [DesignRule(**payload) for payload in self.store.list("design_rules")]

    def _encoder_metrics(self) -> dict[str, dict[str, Any]]:
        by_encoder: dict[str, dict[str, Any]] = {}
        claims_by_run = self._claims_by_run()
        traces_by_run = self._traces_by_run()
        for artifact in self.store.list("artifacts"):
            metrics = artifact.get("metrics", {})
            encoder_type = metrics.get("encoder_type")
            if not encoder_type or "spectral_separability" not in metrics:
                continue
            current = by_encoder.get(encoder_type)
            candidate = {
                "encoder_type": encoder_type,
                "run_id": artifact.get("run_id"),
                "artifact_id": artifact["artifact_id"],
                "trace_ids": traces_by_run.get(artifact.get("run_id"), []),
                "claim_ids": claims_by_run.get(artifact.get("run_id"), []),
                "metrics": metrics,
            }
            if current is None or self._joint(candidate) > self._joint(current):
                by_encoder[encoder_type] = candidate
        return by_encoder

    def _claims_by_run(self) -> dict[str, list[str]]:
        mapping: dict[str, list[str]] = {}
        for claim in self.store.list("claims"):
            run_id = claim.get("scope", {}).get("run_id")
            if run_id:
                mapping.setdefault(run_id, []).append(claim["claim_id"])
        return mapping

    def _traces_by_run(self) -> dict[str, list[str]]:
        mapping: dict[str, list[str]] = {}
        for trace in self.store.list("meta_traces"):
            mapping.setdefault(trace["run_id"], []).append(trace["trace_id"])
        return mapping

    def _joint(self, item: dict[str, Any]) -> float:
        metrics = item["metrics"]
        return round(
            0.35 * float(metrics.get("psf_depth_similarity", 0.0))
            + 0.35 * float(metrics.get("spectral_separability", 0.0))
            + 0.15 * float(metrics.get("mock_mtf_mean", 0.0))
            + 0.15 * float(metrics.get("mock_energy_efficiency", 0.0)),
            6,
        )


class DesignRuleMemory(DesignRuleManager):
    """Backward-compatible alias."""


def compile_rules_from_hsi_matrix(matrix_result: dict[str, Any]) -> list[DesignRule]:
    rows = [row for row in matrix_result.get("rows", []) if row.get("status") == "succeeded"]
    artifact_ids = list(matrix_result.get("artifact_ids", []))
    claim_ids = list(matrix_result.get("claim_ids", []))
    supported_by = sorted(set([*artifact_ids, *claim_ids])) or [str(matrix_result.get("matrix_id", "hsi_matrix"))]
    rules: list[DesignRule] = []

    linear_best = _best_matrix_encoder(rows, "optical_conditioned_linear")
    tiny_best = _best_matrix_encoder(rows, "tiny_cnn")
    if linear_best == "achromatic" and tiny_best in {"chromatic_coded", "controlled_chromatic_edof"}:
        statement = "Chromatic coding benefit is reconstruction-dependent under current synthetic setting."
        rules.append(
            _matrix_rule(
                statement,
                matrix_result,
                supported_by,
                confidence=0.82,
                valid_conditions={
                    "linear_best": linear_best,
                    "tiny_cnn_best": tiny_best,
                    "caveat": "Synthetic/mock matrix only; not a real camera or native DeepLens validation.",
                    "source_artifact_ids": artifact_ids,
                },
            )
        )

    if _controlled_consistently_beats_conventional(rows):
        statement = "Controlled chromatic EDOF improves over conventional baseline under synthetic optical-sensitive HSI evaluation."
        rules.append(
            _matrix_rule(
                statement,
                matrix_result,
                supported_by,
                confidence=0.78,
                valid_conditions={
                    "comparison": "controlled_chromatic_edof rank better than conventional in available groups",
                    "caveat": "Only supported inside the matrix dataset/backend/reconstructor scope.",
                    "source_artifact_ids": artifact_ids,
                },
            )
        )

    if _encoder_rankings_vary(rows):
        statement = "Optical encoder choice affects HSI reconstruction ranking under optical-sensitive forward model."
        rules.append(
            _matrix_rule(
                statement,
                matrix_result,
                supported_by,
                confidence=0.8,
                valid_conditions={
                    "comparison": "available encoder scores or ranks differ within at least one group",
                    "caveat": "Ranking is scoped to the matrix dataset, backend, forward model, and reconstructor.",
                    "source_artifact_ids": artifact_ids,
                },
            )
        )
    return rules


def _matrix_rule(statement: str, matrix_result: dict[str, Any], supported_by: list[str], confidence: float, valid_conditions: dict[str, Any]) -> DesignRule:
    scope = sorted(
        {
            "hsi_matrix",
            *[str(row.get("dataset")) for row in matrix_result.get("rows", []) if row.get("dataset")],
            *[str(row.get("backend")) for row in matrix_result.get("rows", []) if row.get("backend")],
            *[str(row.get("reconstructor")) for row in matrix_result.get("rows", []) if row.get("reconstructor")],
        }
    )
    return DesignRule(
        rule_id=make_deterministic_id("rule", statement, matrix_result.get("matrix_id")),
        statement=statement,
        scope=scope,
        status="active",
        confidence=confidence,
        supported_by=supported_by,
        contradicted_by=[],
        valid_conditions={
            **valid_conditions,
            "matrix_id": matrix_result.get("matrix_id"),
        },
        invalid_at=None,
        superseded_by=None,
        source_trace_ids=list(matrix_result.get("source_trace_ids", [])),
    )


def _best_matrix_encoder(rows: list[dict[str, Any]], reconstructor: str) -> str | None:
    candidates = [row for row in rows if row.get("reconstructor") == reconstructor]
    if not candidates:
        return None
    return min(candidates, key=lambda row: row.get("rank_within_group") or 999).get("encoder")


def _controlled_consistently_beats_conventional(rows: list[dict[str, Any]]) -> bool:
    grouped: dict[tuple[Any, Any, Any, Any], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row.get("dataset"), row.get("backend"), row.get("reconstructor"), row.get("forward_mode")), []).append(row)
    comparisons = []
    for group in grouped.values():
        controlled = next((row for row in group if row.get("encoder") == "controlled_chromatic_edof"), None)
        conventional = next((row for row in group if row.get("encoder") == "conventional"), None)
        if controlled and conventional:
            comparisons.append((controlled.get("rank_within_group") or 999) < (conventional.get("rank_within_group") or 999))
    return bool(comparisons) and all(comparisons)


def _encoder_rankings_vary(rows: list[dict[str, Any]]) -> bool:
    grouped: dict[tuple[Any, Any, Any, Any], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row.get("dataset"), row.get("backend"), row.get("reconstructor"), row.get("forward_mode")), []).append(row)
    for group in grouped.values():
        ranks = {row.get("rank_within_group") for row in group if row.get("rank_within_group") is not None}
        scores = [float(row.get("reconstruction_score") or 0.0) for row in group]
        if len(ranks) > 1 or (scores and max(scores) - min(scores) > 0.05):
            return True
    return False
