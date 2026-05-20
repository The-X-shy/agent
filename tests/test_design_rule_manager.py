from optiresearch.memory.claim_evidence import ClaimEvidenceManager
from optiresearch.memory.design_rule import DesignRuleManager
from optiresearch.runtime.baselines import run_baseline_batch
from optiresearch.storage.sqlite_store import SQLiteStore


def test_compile_design_rule_from_baseline_claims(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "memory.sqlite"))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    report = run_baseline_batch("Design depth-invariant and spectrally discriminative EDOF-HSI encoder", output_root=tmp_path / "baselines")
    manager = DesignRuleManager(SQLiteStore(tmp_path / "memory.sqlite"))

    rules = manager.compile_from_claims()
    rule = next(item for item in rules if "controlled chromatic EDOF" in item.statement)
    explanation = manager.explain_rule(rule.rule_id)

    assert report["best_joint_tradeoff"]["encoder_type"] == "controlled_chromatic_edof"
    assert rule.status == "active"
    assert rule.supported_by
    assert explanation["rule_id"] == rule.rule_id
    assert explanation["evidence"]


def test_detect_contradiction_marks_achromatic_best_claim(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "memory.sqlite"))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    run_baseline_batch("Design depth-invariant and spectrally discriminative EDOF-HSI encoder", output_root=tmp_path / "baselines")
    store = SQLiteStore(tmp_path / "memory.sqlite")
    claim_manager = ClaimEvidenceManager(store)
    old_claim = claim_manager.create_claim(
        "achromatic encoder is best for spectral separability",
        scope={"backend": "mock_deeplens"},
    )
    manager = DesignRuleManager(store)

    contradictions = manager.detect_contradictions()
    updated = claim_manager.get_claim(old_claim.claim_id)

    assert contradictions
    assert updated.status in {"contradicted", "partially_supported"}
    assert updated.contradict_edges


def test_supersede_rule_updates_status(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "memory.sqlite"))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    run_baseline_batch("Design depth-invariant and spectrally discriminative EDOF-HSI encoder", output_root=tmp_path / "baselines")
    manager = DesignRuleManager(SQLiteStore(tmp_path / "memory.sqlite"))
    rule = manager.compile_from_claims()[0]

    updated = manager.supersede_rule(rule.rule_id, "replacement_rule")

    assert updated.status == "superseded"
    assert updated.superseded_by == "replacement_rule"
