from pathlib import Path

from optiresearch.cli import main
from optiresearch.memory.claim_evidence import ClaimEvidenceManager
from optiresearch.memory.design_rule import DesignRuleManager
from optiresearch.runtime.baselines import run_baseline_batch
from optiresearch.reports.paper import export_evidence_tables
from optiresearch.storage.sqlite_store import SQLiteStore


def test_export_paper_summary_and_evidence_tables(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "memory.sqlite"))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("OPTIRESEARCH_BASELINE_ROOT", str(tmp_path / "baselines"))
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))

    run_baseline_batch("Design depth-invariant and spectrally discriminative EDOF-HSI encoder")
    rule = DesignRuleManager(SQLiteStore()).compile_from_claims()[0]
    claim = ClaimEvidenceManager(SQLiteStore()).list_claims()[0]

    main(["export-paper-summary"])
    main(["export-evidence-tables"])

    summary_path = tmp_path / "reports" / "phase3_experiment_summary.md"
    claims_path = tmp_path / "reports" / "evidence_claims.md"
    rules_path = tmp_path / "reports" / "evidence_rules.md"

    assert summary_path.exists()
    assert claims_path.exists()
    assert rules_path.exists()
    assert "Baseline Comparison" in summary_path.read_text(encoding="utf-8")
    assert claim.claim_id in claims_path.read_text(encoding="utf-8")
    assert rule.rule_id in rules_path.read_text(encoding="utf-8")


def test_export_evidence_tables_handles_legacy_claim_edges(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "legacy.sqlite"))
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))
    store = SQLiteStore()
    store.init_db()
    store.upsert(
        "claims",
        "claim_legacy",
        {
            "claim_id": "claim_legacy",
            "text": "legacy claim",
            "status": "supported",
            "support_score": 0.9,
            "support_edges": [{"artifact_id": "artifact_legacy", "relation": "supports", "score": 0.9}],
            "contradict_edges": [],
            "scope": {},
            "review_status": "reviewed",
            "required_caveats": [],
            "warnings": [],
        },
        workspace_id="default",
    )

    paths = export_evidence_tables(store=store)

    assert paths["claims"].exists()
    assert "claim_legacy" in paths["claims"].read_text(encoding="utf-8")
