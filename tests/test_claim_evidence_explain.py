from optiresearch.memory.claim_evidence import ClaimEvidenceManager
from optiresearch.runtime.graph import run_mvp_flow
from optiresearch.storage.sqlite_store import SQLiteStore


def test_explain_claim_returns_metric_artifact_and_trace(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "memory.sqlite"))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    result = run_mvp_flow("Design a mock EDOF-HSI optical encoder", workspace_id="opti_lab")
    store = SQLiteStore(tmp_path / "memory.sqlite")
    manager = ClaimEvidenceManager(store, workspace_id="opti_lab")
    depth_claim_id = result["claims"][0]["claim_id"]

    explanation = manager.explain_claim(depth_claim_id)

    assert explanation["claim_text"].startswith("The mock optical encoder")
    assert explanation["evidence_table"][0]["artifact_id"]
    assert explanation["evidence_table"][0]["metric_name"] == "psf_depth_similarity"
    assert explanation["evidence_table"][0]["metric_value"] >= 0.8
    assert explanation["source_traces"]
    assert "mock-backed evidence only" in explanation["caveats"]
