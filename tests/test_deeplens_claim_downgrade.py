from optiresearch.memory.claim_evidence import (
    DEEPLENS_SMOKE_CAVEAT,
    ClaimEvidenceManager,
)
from optiresearch.storage.sqlite_store import SQLiteStore


def test_smoke_level_deeplens_encoder_claim_is_not_supported(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "claims.sqlite"))
    store = SQLiteStore()
    manager = ClaimEvidenceManager(store)
    claim = manager.create_claim(
        "controlled chromatic EDOF improves spectral separability under real DeepLens",
        scope={
            "backend": "deeplens",
            "backend_capability_level": "smoke",
            "encoder_behavior_realized": False,
        },
    )
    manager.attach_support(claim.claim_id, "artifact_metric", 0.95)

    reviewed = manager.review_claim(claim.claim_id)

    assert reviewed.status in {"partially_supported", "needs_followup", "unsupported"}
    assert reviewed.status != "supported"
    assert DEEPLENS_SMOKE_CAVEAT in reviewed.required_caveats


def test_valid_deeplens_psf_artifact_claim_can_be_supported(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "claims.sqlite"))
    store = SQLiteStore()
    manager = ClaimEvidenceManager(store)
    claim = manager.create_claim(
        "DeepLens backend produced valid PSF artifacts",
        scope={
            "backend": "deeplens",
            "backend_capability_level": "smoke",
            "encoder_behavior_realized": False,
        },
    )
    manager.attach_support(claim.claim_id, "artifact_psf", 0.9)

    reviewed = manager.review_claim(claim.claim_id)

    assert reviewed.status == "supported"
    assert DEEPLENS_SMOKE_CAVEAT in reviewed.required_caveats
