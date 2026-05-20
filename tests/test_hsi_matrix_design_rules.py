from optiresearch.memory.design_rule import compile_rules_from_hsi_matrix


def test_compile_rules_from_hsi_matrix_includes_scope_and_caveats():
    matrix_result = {
        "matrix_id": "matrix_rules",
        "artifact_ids": ["artifact_matrix"],
        "claim_ids": ["claim_matrix"],
        "rows": [
            {"dataset": "synthetic", "backend": "mock_deeplens", "encoder": "achromatic", "reconstructor": "optical_conditioned_linear", "rank_within_group": 1, "reconstruction_score": 5.0, "status": "succeeded"},
            {"dataset": "synthetic", "backend": "mock_deeplens", "encoder": "controlled_chromatic_edof", "reconstructor": "optical_conditioned_linear", "rank_within_group": 2, "reconstruction_score": 4.5, "status": "succeeded"},
            {"dataset": "synthetic", "backend": "mock_deeplens", "encoder": "conventional", "reconstructor": "tiny_cnn", "rank_within_group": 3, "reconstruction_score": 3.0, "status": "succeeded"},
            {"dataset": "synthetic", "backend": "mock_deeplens", "encoder": "controlled_chromatic_edof", "reconstructor": "tiny_cnn", "rank_within_group": 1, "reconstruction_score": 6.0, "status": "succeeded"},
        ],
    }

    rules = compile_rules_from_hsi_matrix(matrix_result)

    assert rules
    assert any("reconstruction-dependent" in rule.statement for rule in rules)
    for rule in rules:
        assert rule.supported_by
        assert rule.scope
        assert rule.confidence > 0.0
        assert "caveat" in rule.valid_conditions
        assert rule.valid_conditions["source_artifact_ids"] == ["artifact_matrix"]

