"""Markdown exporters for paper experiment summaries and evidence tables."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from optiresearch.memory.design_rule import DesignRuleManager
from optiresearch.storage.sqlite_store import SQLiteStore


def report_root(path: Optional[Path] = None) -> Path:
    return path or Path(os.getenv("OPTIRESEARCH_REPORT_ROOT", "./workspace/reports"))


def export_phase3_experiment_summary(
    store: Optional[SQLiteStore] = None,
    output_root: Optional[Path] = None,
) -> Path:
    """Write the frozen Phase 3 paper experiment summary markdown."""

    store = store or SQLiteStore()
    store.init_db()
    root = report_root(output_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "phase3_experiment_summary.md"
    path.write_text(_phase3_summary_markdown(store), encoding="utf-8")
    return path


def export_evidence_tables(
    store: Optional[SQLiteStore] = None,
    output_root: Optional[Path] = None,
) -> dict[str, Path]:
    """Write claim and design-rule evidence markdown tables."""

    store = store or SQLiteStore()
    store.init_db()
    root = report_root(output_root)
    root.mkdir(parents=True, exist_ok=True)
    claims_path = root / "evidence_claims.md"
    rules_path = root / "evidence_rules.md"
    claims_path.write_text(_claims_markdown(store), encoding="utf-8")
    rules_path.write_text(_rules_markdown(store), encoding="utf-8")
    return {"claims": claims_path, "rules": rules_path}


def _phase3_summary_markdown(store: SQLiteStore) -> str:
    baseline = _read_json(Path(os.getenv("OPTIRESEARCH_BASELINE_ROOT", "./workspace/baselines")) / "baseline_comparison.json")
    benchmark = _read_json(Path(os.getenv("OPTIRESEARCH_BENCHMARK_ROOT", "./workspace/benchmarks")) / "opti_memory_bench_report.json")
    claims = store.list("claims")
    unsupported = [claim for claim in claims if claim.get("status") == "unsupported"]
    unsupported_rate = round(len(unsupported) / len(claims), 6) if claims else 0.0
    lines = [
        "# Phase 3 Experiment Summary",
        "",
        "## Baseline Comparison",
        "",
        _baseline_table(baseline),
        "",
        "## Memory Ablation",
        "",
        _ablation_table(benchmark),
        "",
        "## Design Rule Compilation",
        "",
        _design_rule_table(store),
        "",
        "## Claim Contradiction",
        "",
        _claim_contradiction_table(claims),
        "",
        "## Unsupported Claim Rate",
        "",
        "| Claim Count | Unsupported Count | Unsupported Rate |",
        "|---:|---:|---:|",
        f"| {len(claims)} | {len(unsupported)} | {unsupported_rate} |",
        "",
    ]
    return "\n".join(lines)


def _baseline_table(baseline: dict[str, Any]) -> str:
    if not baseline:
        return "No baseline comparison has been generated."
    lines = [
        "| Encoder | Run ID | Depth Similarity | Spectral Separability | MTF | Energy | Joint |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in baseline.get("runs", []):
        metrics = item.get("metrics", {})
        lines.append(
            "| {encoder} | {run_id} | {depth} | {spectral} | {mtf} | {energy} | {joint} |".format(
                encoder=_cell(item.get("encoder_type")),
                run_id=_cell(item.get("run_id")),
                depth=_cell(metrics.get("psf_depth_similarity")),
                spectral=_cell(metrics.get("spectral_separability")),
                mtf=_cell(metrics.get("mock_mtf_mean")),
                energy=_cell(metrics.get("mock_energy_efficiency")),
                joint=_cell(item.get("joint_tradeoff_score")),
            )
        )
    best = baseline.get("best_joint_tradeoff", {})
    if best:
        lines.extend(["", f"Best joint tradeoff: `{_cell(best.get('encoder_type'))}`."])
    return "\n".join(lines)


def _ablation_table(benchmark: dict[str, Any]) -> str:
    ablations = benchmark.get("ablations") if benchmark else None
    if not ablations:
        return "No memory ablation benchmark has been generated."
    lines = [
        "| Mode | Plan Hit | Evidence Complete | Unsupported Claim Rate | Trigger Precision | Total Score |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode, metrics in sorted(ablations.items()):
        lines.append(
            "| {mode} | {plan_hit} | {evidence} | {unsupported} | {precision} | {score} |".format(
                mode=_cell(mode),
                plan_hit=_cell(metrics.get("plan_hit")),
                evidence=_cell(metrics.get("evidence_complete")),
                unsupported=_cell(metrics.get("unsupported_claim_rate")),
                precision=_cell(metrics.get("trigger_precision")),
                score=_cell(metrics.get("total_score")),
            )
        )
    return "\n".join(lines)


def _design_rule_table(store: SQLiteStore) -> str:
    rules = DesignRuleManager(store).list_rules()
    if not rules:
        return "No design rules have been compiled."
    lines = [
        "| Rule ID | Status | Confidence | Statement | Supported By | Contradicted By |",
        "|---|---|---:|---|---|---|",
    ]
    for rule in rules:
        lines.append(
            "| {rule_id} | {status} | {confidence} | {statement} | {supported} | {contradicted} |".format(
                rule_id=_cell(rule.rule_id),
                status=_cell(rule.status),
                confidence=_cell(rule.confidence),
                statement=_cell(rule.statement),
                supported=_cell(", ".join(rule.supported_by)),
                contradicted=_cell(", ".join(rule.contradicted_by)),
            )
        )
    return "\n".join(lines)


def _claim_contradiction_table(claims: list[dict[str, Any]]) -> str:
    rows = [claim for claim in claims if claim.get("contradict_edges") or claim.get("status") == "contradicted"]
    if not rows:
        return "No contradicted claims have been detected."
    lines = [
        "| Claim ID | Status | Claim | Contradicting Artifact | Metric | Value | Score |",
        "|---|---|---|---|---|---:|---:|",
    ]
    for claim in rows:
        edges = claim.get("contradict_edges") or [{}]
        for edge in edges:
            lines.append(
                "| {claim_id} | {status} | {text} | {artifact} | {metric} | {value} | {score} |".format(
                    claim_id=_cell(claim.get("claim_id")),
                    status=_cell(claim.get("status")),
                    text=_cell(claim.get("text")),
                    artifact=_cell(edge.get("artifact_id")),
                    metric=_cell(edge.get("metric_name")),
                    value=_cell(edge.get("metric_value")),
                    score=_cell(edge.get("score")),
                )
            )
    return "\n".join(lines)


def _claims_markdown(store: SQLiteStore) -> str:
    claims = store.list("claims")
    lines = [
        "# Claim Evidence Table",
        "",
        "| Claim ID | Status | Artifact ID | Evidence Role | Artifact Type | SHA256 | Remote Job | Artifact Binding | Claim | Score | Caveats |",
        "|---|---|---|---|---|---|---|---|---:|---|",
    ]
    for claim in claims:
        rows = [*_normalize_edges(claim.get("support_edges", [])), *_normalize_edges(claim.get("contradict_edges", []))]
        if not rows:
            # Show no_artifact_binding row
            lines.append(
                "| {claim_id} | {status} | {artifact} | {role} | {atype} | {sha} | {job} | {binding} | {text} | {score} | {caveats} |".format(
                    claim_id=_cell(claim.get("claim_id")),
                    status=_cell(claim.get("status")),
                    artifact="no_artifact_binding",
                    role="",
                    atype="",
                    sha="",
                    job="",
                    binding="no_artifact_binding",
                    text=_cell(claim.get("text")),
                    score="",
                    caveats=_cell("; ".join(claim.get("required_caveats", []))),
                )
            )
        for edge in rows:
            aid = edge.get("artifact_id", "")
            lines.append(
                "| {claim_id} | {status} | {artifact} | {role} | {atype} | {sha} | {job} | {binding} | {text} | {score} | {caveats} |".format(
                    claim_id=_cell(claim.get("claim_id")),
                    status=_cell(claim.get("status")),
                    artifact=_cell(aid or "no_artifact_binding"),
                    role=_cell(edge.get("evidence_role", "")),
                    atype=_cell(edge.get("artifact_type", "")),
                    sha=_cell((edge.get("artifact_sha256", "") or "")[:16]),
                    job=_cell(edge.get("remote_job_id", "")),
                    binding="bound" if aid else "no_artifact_binding",
                    text=_cell(claim.get("text")),
                    score=_cell(edge.get("score")),
                    caveats=_cell("; ".join(claim.get("required_caveats", []))),
                )
            )
    return "\n".join(lines) + "\n"


def _rules_markdown(store: SQLiteStore) -> str:
    manager = DesignRuleManager(store)
    rules = manager.list_rules()
    lines = [
        "# Design Rule Evidence Table",
        "",
        "| Rule ID | Status | Confidence | Statement | Evidence Type | Evidence ID | Evidence Detail | Source Traces |",
        "|---|---|---:|---|---|---|---|---|",
    ]
    for rule in rules:
        explanation = manager.explain_rule(rule.rule_id)
        evidence_rows = explanation["evidence"] or [{}]
        for evidence in evidence_rows:
            lines.append(
                "| {rule_id} | {status} | {confidence} | {statement} | {etype} | {eid} | {detail} | {traces} |".format(
                    rule_id=_cell(rule.rule_id),
                    status=_cell(rule.status),
                    confidence=_cell(rule.confidence),
                    statement=_cell(rule.statement),
                    etype=_cell(evidence.get("type")),
                    eid=_cell(evidence.get("id")),
                    detail=_cell(_evidence_detail(evidence)),
                    traces=_cell(", ".join(rule.source_trace_ids)),
                )
            )
    return "\n".join(lines) + "\n"


def _evidence_detail(evidence: dict[str, Any]) -> str:
    if not evidence:
        return ""
    if evidence.get("type") == "artifact":
        metrics = evidence.get("metrics", {})
        keys = ["encoder_type", "psf_depth_similarity", "spectral_separability", "mock_mtf_mean", "mock_energy_efficiency"]
        return ", ".join(f"{key}={metrics.get(key)}" for key in keys if key in metrics)
    if evidence.get("type") == "claim":
        return f"{evidence.get('status')}: {evidence.get('text')}"
    return json.dumps(evidence, ensure_ascii=False, sort_keys=True)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")


def _normalize_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for edge in edges:
        normalized.append(
            {
                "artifact_id": edge.get("artifact_id"),
                "trace_id": edge.get("trace_id"),
                "metric_name": edge.get("metric_name"),
                "metric_value": edge.get("metric_value"),
                "relation": edge.get("relation"),
                "score": edge.get("score"),
                "rationale": edge.get("rationale", ""),
            }
        )
    return normalized
