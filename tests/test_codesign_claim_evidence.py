"""Test ClaimEvidence for co-design optimization."""
from optiresearch.memory.claim_evidence import ClaimEvidenceManager
from optiresearch.schemas.optimization import build_default_optimization_spec
from optiresearch.runtime.codesign_loop import run_codesign_loop


def test_codesign_claim_has_evidence_level(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))
    monkeypatch.setenv("OPTIRESEARCH_HSI_ROOT", str(tmp_path / "hsi"))
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    spec = build_default_optimization_spec(["PSNR"], backend="mock_deeplens", objective="test_claim")
    spec.max_iterations = 1
    spec.psf_source = "parameterized_mock"
    spec.llm_provider = "mock"

    result = run_codesign_loop(spec)

    manager = ClaimEvidenceManager(workspace_id=result["loop_id"])
    claims = manager.list_claims()
    assert len(claims) >= 1, f"No claims found for workspace {result['loop_id']}"


def test_mock_psf_claim_has_mock_caveat(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))
    monkeypatch.setenv("OPTIRESEARCH_HSI_ROOT", str(tmp_path / "hsi"))
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    spec = build_default_optimization_spec(["PSNR"], backend="mock_deeplens", objective="test_caveat_claim")
    spec.max_iterations = 1
    spec.psf_source = "parameterized_mock"
    spec.llm_provider = "mock"

    result = run_codesign_loop(spec)

    manager = ClaimEvidenceManager(workspace_id=result["loop_id"])
    for c in manager.list_claims():
        explanation = manager.explain_claim(c.claim_id)
        assert "evidence_level" in explanation
