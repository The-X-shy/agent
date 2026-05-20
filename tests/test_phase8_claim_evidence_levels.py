from optiresearch.memory.claim_evidence import ClaimEvidenceManager
from optiresearch.storage.sqlite_store import SQLiteStore


def test_semi_native_claim_scope_is_supported_but_native_claim_is_not(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "claims.sqlite"))
    manager = ClaimEvidenceManager(SQLiteStore())
    semi = manager.create_claim(
        "conventional baseline is supported under semi-native DeepLens ParaxialLens behavior",
        scope={
            "backend": "deeplens",
            "selected_realization_level": "semi_native",
            "semi_native_succeeded": True,
            "claim_scope": "baseline DeepLens ParaxialLens behavior",
        },
    )
    manager.attach_support(semi.claim_id, "artifact", 0.9)
    native = manager.create_claim(
        "controlled chromatic EDOF is native physically optimized under DeepLens",
        scope={
            "backend": "deeplens",
            "selected_realization_level": "semi_native",
            "semi_native_succeeded": True,
        },
    )
    manager.attach_support(native.claim_id, "artifact", 0.95)

    semi_reviewed = manager.review_claim(semi.claim_id)
    native_reviewed = manager.review_claim(native.claim_id)
    explanation = manager.explain_claim(semi.claim_id)

    assert semi_reviewed.status == "supported"
    assert semi_reviewed.metadata["evidence_level"] == "deeplens_semi_native"
    assert native_reviewed.status == "needs_followup"
    assert explanation["evidence_level"] == "deeplens_semi_native"
    assert explanation["allowed_claim_scope"] == "baseline DeepLens ParaxialLens behavior"
