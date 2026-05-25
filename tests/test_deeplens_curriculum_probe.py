from optiresearch.runtime.deeplens_curriculum_probe import (
    run_deeplens_curriculum_probe,
)


def test_curriculum_probe_returns_structured_result():
    result = run_deeplens_curriculum_probe(max_steps=2, device="cpu")
    assert "status" in result
    assert "evidence_level" in result
    assert "stages_completed" in result
    assert result["evidence_level"] == "diagnostic_evidence"


def test_curriculum_probe_completes_quickly():
    import time
    start = time.perf_counter()
    result = run_deeplens_curriculum_probe(max_steps=2, device="cpu")
    elapsed = time.perf_counter() - start
    assert result["status"] == "succeeded"
    assert elapsed < 30.0
