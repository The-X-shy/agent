from optiresearch.memory.claim_evidence import ClaimEvidenceManager
from optiresearch.storage.sqlite_store import SQLiteStore


def test_claim_without_support_is_unsupported(tmp_path):
    store = SQLiteStore(tmp_path / "memory.sqlite")
    store.init_db()
    manager = ClaimEvidenceManager(store, workspace_id="ws")

    claim = manager.create_claim("The encoder is stable.", scope={"backend": "mock_deeplens"})
    reviewed = manager.review_claim(claim.claim_id)

    assert reviewed.status == "unsupported"
    assert reviewed.support_score == 0.0
    assert "currently simulation-only or mock-backed" in reviewed.required_caveats


def test_claim_with_high_support_is_supported(tmp_path):
    store = SQLiteStore(tmp_path / "memory.sqlite")
    store.init_db()
    manager = ClaimEvidenceManager(store, workspace_id="ws")

    claim = manager.create_claim("The encoder is stable.", scope={"backend": "mock_deeplens"})
    manager.attach_support(claim.claim_id, "artifact-1", 0.82)
    reviewed = manager.review_claim(claim.claim_id)

    assert reviewed.status == "supported"
    assert reviewed.support_score == 0.82


def test_claim_with_stronger_contradiction_is_contradicted(tmp_path):
    store = SQLiteStore(tmp_path / "memory.sqlite")
    store.init_db()
    manager = ClaimEvidenceManager(store, workspace_id="ws")

    claim = manager.create_claim("The encoder is stable.", scope={"backend": "mock_deeplens"})
    manager.attach_support(claim.claim_id, "artifact-1", 0.6)
    manager.attach_contradiction(claim.claim_id, "artifact-2", 0.9)
    reviewed = manager.review_claim(claim.claim_id)

    assert reviewed.status == "contradicted"
    assert reviewed.support_score == 0.6
