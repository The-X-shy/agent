from optiresearch.memory.claim_evidence import ClaimEvidenceManager
from optiresearch.runtime.hsi_matrix import evaluate_matrix_claims


def test_matrix_level_claims_distinguish_reconstructor_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "memory.sqlite"))
    manager = ClaimEvidenceManager()
    matrix_result = {
        "matrix_id": "matrix_test",
        "artifact_ids": ["artifact_matrix"],
        "rows": [
            {"dataset": "synthetic", "backend": "mock_deeplens", "encoder": "achromatic", "reconstructor": "optical_conditioned_linear", "rank_within_group": 1, "status": "succeeded"},
            {"dataset": "synthetic", "backend": "mock_deeplens", "encoder": "conventional", "reconstructor": "optical_conditioned_linear", "rank_within_group": 2, "status": "succeeded"},
            {"dataset": "synthetic", "backend": "mock_deeplens", "encoder": "controlled_chromatic_edof", "reconstructor": "tiny_cnn", "rank_within_group": 1, "status": "succeeded"},
            {"dataset": "synthetic", "backend": "mock_deeplens", "encoder": "conventional", "reconstructor": "tiny_cnn", "rank_within_group": 3, "status": "succeeded"},
        ],
    }

    claims = evaluate_matrix_claims(matrix_result, manager)
    by_text = {claim.text: claim for claim in claims}
    explanation = manager.explain_claim(by_text["chromatic coding benefits require stronger reconstruction network"].claim_id)

    assert by_text["controlled chromatic EDOF improves synthetic HSI reconstruction with tiny CNN"].status == "supported"
    assert by_text["achromatic remains best across all reconstructors"].status in {"contradicted", "partially_supported"}
    assert by_text["chromatic coding benefits require stronger reconstruction network"].status == "supported"
    assert explanation["matrix_evidence"]["matrix_id"] == "matrix_test"
    assert explanation["rank_comparison"]

