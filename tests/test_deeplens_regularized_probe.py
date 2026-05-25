from optiresearch.runtime.deeplens_regularized_probe import (
    run_deeplens_regularized_probe,
)


def test_regularized_probe_returns_structured_result():
    result = run_deeplens_regularized_probe(max_steps=2, device="cpu")
    assert "status" in result
    assert "evidence_level" in result
    assert "base_loss" in result
    assert result["evidence_level"] == "diagnostic_evidence"


def test_regularized_probe_has_reg_terms():
    result = run_deeplens_regularized_probe(max_steps=2, device="cpu")
    assert result["status"] == "succeeded"
    assert result["reg_terms"]
    assert "energy" in result["reg_terms"]
