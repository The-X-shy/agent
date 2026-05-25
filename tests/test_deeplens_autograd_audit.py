from optiresearch.runtime.deeplens_autograd_audit import run_deeplens_autograd_audit


def test_autograd_audit_returns_structured_result():
    result = run_deeplens_autograd_audit(device="cpu")
    assert "status" in result
    assert "trainable_param_count" in result
    assert "evidence_level" in result
    assert result["evidence_level"] == "diagnostic_evidence"


def test_autograd_audit_handles_missing_deeplens():
    result = run_deeplens_autograd_audit(lens_file="nonexistent_lens", device="cpu")
    assert result["status"] in ("needs_followup", "unavailable")
