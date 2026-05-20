import os

import pytest

from optiresearch.runtime.hsi_pipeline import run_hsi_reconstruction_flow


@pytest.mark.skipif(
    os.getenv("OPTIRESEARCH_ENABLE_REAL_DEEPLENS_TESTS") != "1",
    reason="Real DeepLens HSI pipeline requires explicit opt-in.",
)
def test_real_deeplens_hsi_pipeline_marks_evidence_level(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "memory.sqlite"))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    result = run_hsi_reconstruction_flow(
        "Evaluate synthetic HSI reconstruction with DeepLens encoder",
        backend="deeplens",
        encoder_type="controlled_chromatic_edof",
        realization="auto",
    )

    assert result["metrics"]["PSNR"] > 0.0
    assert result["evidence_level"] in {"hsi_reconstruction_deeplens_proxy", "hsi_reconstruction_deeplens_semi_native"}
