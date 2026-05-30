"""Tests that claim policy matrix blocks overclaims."""
from __future__ import annotations

from optiresearch.system.claim_policy_matrix import generate_claim_policy_matrix


def test_low_levels_block_high_claims():
    matrix = generate_claim_policy_matrix()
    low_levels = ["unsupported", "mock_simulation", "deeplens_integration_smoke"]
    for row in matrix["rows"]:
        if row["evidence_level"] in low_levels:
            assert len(row["blocked_claims"]) > 0, \
                f"Level {row['evidence_level']} should have blocked claims"


def test_high_level_has_fewer_blocked_claims():
    matrix = generate_claim_policy_matrix()
    low = next(r for r in matrix["rows"] if r["evidence_level"] == "native_component_optimization")
    high = next(r for r in matrix["rows"] if r["evidence_level"] == "rollback_protected_native_lens_hsi")
    assert len(low["blocked_claims"]) > len(high["blocked_claims"])


def test_real_hsi_level_has_no_blocked_claims():
    matrix = generate_claim_policy_matrix()
    real = next(r for r in matrix["rows"] if r["evidence_level"] == "real_hsi_performance")
    assert len(real["blocked_claims"]) == 0


def test_all_levels_have_safe_wording():
    matrix = generate_claim_policy_matrix()
    for row in matrix["rows"]:
        assert row["safe_wording_template"], \
            f"Level {row['evidence_level']} should have safe_wording_template"


def test_synthetic_levels_block_real_claims():
    matrix = generate_claim_policy_matrix()
    synthetic_levels = [
        "native_lens_simulation", "native_waveoptics_simulation",
        "stable_native_lens_hsi_codesign", "component_surrogate_hsi_codesign",
    ]
    for row in matrix["rows"]:
        if row["evidence_level"] in synthetic_levels:
            blocked_text = " ".join(row["blocked_claims"]).lower()
            assert "real hsi" in blocked_text or "physical" in blocked_text, \
                f"Level {row['evidence_level']} should block real HSI claims"
