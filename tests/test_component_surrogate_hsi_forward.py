"""Tests for component surrogate HSI forward co-design loop."""

import pytest

from optiresearch.hsi.component_surrogate_forward import run_component_surrogate_hsi_forward
from optiresearch.schemas.component_surrogate_psf import ComponentSurrogateHSICoDesignSpec


@pytest.mark.parametrize("component", ["fresnel", "binary2phase"])
def test_component_surrogate_hsi_forward_updates_component(component, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = ComponentSurrogateHSICoDesignSpec(
        component_type=component,
        steps=3,
        band_count=4,
        image_size=16,
        psf_size=9,
        batch_size=1,
        device="cpu",
    )

    result = run_component_surrogate_hsi_forward(spec)

    assert result.status == "succeeded"
    assert result.reconstruction_loss_before is not None
    assert result.reconstruction_loss_after is not None
    assert result.mse_before is not None and result.mse_after is not None
    assert result.psnr_before is not None and result.psnr_after is not None
    assert result.sam_before is not None and result.sam_after is not None
    assert result.component_grad_norm_max > 0
    assert result.component_parameter_changed is True
    assert result.psf_requires_grad is True
    assert result.loss_requires_grad is True
    assert result.evidence_level == "component_surrogate_hsi_codesign"
    assert result.claim_ceiling == "component_surrogate_hsi_codesign"
    assert result.reconstruction_loss_after <= result.reconstruction_loss_before
    assert "result.json" in " ".join(result.artifacts)
    assert (tmp_path / "workspace" / "component_surrogate_hsi" / result.run_id / "metrics.json").exists()
    assert (tmp_path / "workspace" / "component_surrogate_hsi" / result.run_id / "psf_artifact.npz").exists()


def test_component_surrogate_hsi_rejects_real_dataset():
    spec = ComponentSurrogateHSICoDesignSpec(component_type="fresnel", dataset="real_hsi")

    result = run_component_surrogate_hsi_forward(spec)

    assert result.status == "needs_followup"
    assert result.error_code == "UNSUPPORTED_DATASET"
    assert result.evidence_level == "diagnostic_evidence"
