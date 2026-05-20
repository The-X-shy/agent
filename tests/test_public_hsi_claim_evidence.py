from optiresearch.memory.claim_evidence import ClaimEvidenceManager


def test_public_hsi_claim_requires_public_matrix_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "memory.sqlite"))
    manager = ClaimEvidenceManager()
    claim = manager.create_claim(
        "controlled chromatic EDOF improves public HSI reconstruction",
        scope={
            "evidence_domain": "public_hsi_matrix",
            "dataset_family": "local_npz",
            "dataset_manifest_id": "artifact_dataset",
            "backend": "mock_deeplens",
            "reconstructor": "optical_conditioned_linear",
            "matrix_result": {
                "rows": [
                    {"encoder": "controlled_chromatic_edof", "rank_within_group": 1},
                    {"encoder": "conventional", "rank_within_group": 2},
                ]
            },
        },
    )

    reviewed = manager.review_claim(claim.claim_id)
    explanation = manager.explain_claim(claim.claim_id)

    assert reviewed.status == "supported"
    assert explanation["dataset_family"] == "local_npz"
    assert explanation["optical_backend_evidence_level"] == "public_hsi_mock"


def test_real_camera_public_hsi_claim_is_unsupported_without_real_camera():
    manager = ClaimEvidenceManager()
    claim = manager.create_claim(
        "controlled chromatic EDOF improves real camera HSI",
        scope={"evidence_domain": "public_hsi_matrix", "dataset_family": "local_npz", "backend": "mock_deeplens"},
    )

    reviewed = manager.review_claim(claim.claim_id)

    assert reviewed.status == "unsupported"
    assert "real_camera_evidence_missing" in reviewed.warnings


def test_public_dataset_mock_optical_design_claim_is_partially_supported():
    manager = ClaimEvidenceManager()
    claim = manager.create_claim(
        "public dataset result validates optical design",
        scope={"evidence_domain": "public_hsi_matrix", "dataset_family": "local_npz", "backend": "mock_deeplens", "dataset_manifest_id": "artifact_dataset", "matrix_id": "matrix"},
    )

    reviewed = manager.review_claim(claim.claim_id)

    assert reviewed.status == "partially_supported"
    assert "public_data_but_mock_optical_measurement" in reviewed.warnings

