from optiresearch.memory.claim_evidence import ClaimEvidenceManager
from optiresearch.storage.file_artifact_store import FileArtifactStore
from optiresearch.storage.sqlite_store import SQLiteStore


def test_hsi_reconstruction_claim_requires_reconstruction_metrics(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "memory.sqlite"))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    store = SQLiteStore()
    artifact_store = FileArtifactStore(root=tmp_path / "artifacts", store=store)
    ref = artifact_store.register_json(
        {"PSNR": 22.0, "SAM": 0.2, "encoder_type": "controlled_chromatic_edof", "baseline_comparison": "mock"},
        workspace_id="default",
        run_id="run_hsi",
        trace_id=None,
        producer="HSIReconstructionPipeline",
        metadata={"filename": "reconstruction_metrics.json", "artifact_type": "metrics", "backend": "mock_deeplens"},
        metrics={"PSNR": 22.0, "SAM": 0.2},
    )
    manager = ClaimEvidenceManager(store)
    claim = manager.create_claim(
        "controlled chromatic EDOF improves HSI reconstruction under mock setting",
        scope={"backend": "mock_deeplens", "run_id": "run_hsi", "evidence_domain": "hsi_reconstruction"},
    )

    reviewed = manager.review_claim(claim.claim_id)
    explanation = manager.explain_claim(claim.claim_id)

    assert reviewed.status == "supported"
    assert explanation["evidence_level"] == "hsi_reconstruction_mock"
    assert reviewed.support_edges[0].artifact_id == ref.artifact_id


def test_real_hsi_reconstruction_claim_needs_followup_without_native_pipeline(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "memory.sqlite"))
    manager = ClaimEvidenceManager(SQLiteStore())
    claim = manager.create_claim(
        "controlled chromatic EDOF improves real HSI reconstruction",
        scope={"backend": "deeplens", "evidence_domain": "hsi_reconstruction", "selected_realization_level": "adapter_proxy"},
    )
    manager.attach_support(claim.claim_id, "artifact", 0.9)

    reviewed = manager.review_claim(claim.claim_id)

    assert reviewed.status in {"needs_followup", "unsupported", "partially_supported"}
    assert reviewed.status != "supported"
