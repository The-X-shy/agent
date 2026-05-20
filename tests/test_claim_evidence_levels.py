from optiresearch.memory.claim_evidence import ClaimEvidenceManager
from optiresearch.storage.sqlite_store import SQLiteStore


def test_adapter_proxy_encoder_claim_can_be_supported_with_proxy_caveat(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "claims.sqlite"))
    manager = ClaimEvidenceManager(SQLiteStore())
    claim = manager.create_claim(
        "controlled chromatic EDOF improves joint depth-spectral tradeoff under adapter-proxy DeepLens setting",
        scope={
            "backend": "deeplens",
            "backend_capability_level": "proxy",
            "encoder_behavior_realized": True,
            "encoder_behavior_realization_level": "adapter_proxy",
            "physical_validation_level": "deeplens_base_psf_plus_adapter_proxy",
        },
    )
    manager.attach_support(claim.claim_id, "artifact_metrics", 0.91)

    reviewed = manager.review_claim(claim.claim_id)

    assert reviewed.status == "supported"
    assert "adapter-proxy DeepLens evidence; not native physical validation" in reviewed.required_caveats


def test_adapter_proxy_physical_claim_is_not_fully_supported(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "claims.sqlite"))
    manager = ClaimEvidenceManager(SQLiteStore())
    claim = manager.create_claim(
        "controlled chromatic EDOF is physically validated as best under DeepLens",
        scope={
            "backend": "deeplens",
            "backend_capability_level": "proxy",
            "encoder_behavior_realized": True,
            "encoder_behavior_realization_level": "adapter_proxy",
            "physical_validation_level": "deeplens_base_psf_plus_adapter_proxy",
        },
    )
    manager.attach_support(claim.claim_id, "artifact_metrics", 0.95)

    reviewed = manager.review_claim(claim.claim_id)

    assert reviewed.status in {"partially_supported", "needs_followup"}
    assert reviewed.status != "supported"


def test_adapter_proxy_artifact_claim_can_be_supported(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "claims.sqlite"))
    manager = ClaimEvidenceManager(SQLiteStore())
    claim = manager.create_claim(
        "DeepLens adapter can produce encoder-specific baseline artifacts",
        scope={
            "backend": "deeplens",
            "backend_capability_level": "proxy",
            "encoder_behavior_realized": True,
            "encoder_behavior_realization_level": "adapter_proxy",
        },
    )
    manager.attach_support(claim.claim_id, "artifact_manifest", 0.88)

    reviewed = manager.review_claim(claim.claim_id)

    assert reviewed.status == "supported"
