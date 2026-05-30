"""Tests for claim policy matrix generation."""
from __future__ import annotations

from optiresearch.system.claim_policy_matrix import generate_claim_policy_matrix


def test_generate_claim_policy_matrix():
    matrix = generate_claim_policy_matrix()
    assert matrix["matrix_version"] == "0.1"
    assert matrix["evidence_levels_covered"] > 0
    assert len(matrix["rows"]) == matrix["evidence_levels_covered"]


def test_matrix_rows_have_required_columns():
    matrix = generate_claim_policy_matrix()
    for row in matrix["rows"]:
        assert "evidence_level" in row
        assert "rank" in row
        assert "supported_claims" in row
        assert "blocked_claims" in row
        assert "safe_wording_template" in row
        assert "required_artifacts" in row
        assert "required_metrics" in row
        assert "downgrade_conditions" in row


def test_covered_evidence_levels():
    matrix = generate_claim_policy_matrix()
    levels = {row["evidence_level"] for row in matrix["rows"]}
    assert "unsupported" in levels
    assert "native_lens_simulation" in levels
    assert "real_hsi_performance" in levels
    assert "stable_native_lens_hsi_codesign" in levels
