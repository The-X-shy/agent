"""Runtime handler tests for component surrogate HSI co-design."""

from optiresearch.runtime.component_surrogate_hsi_codesign import run_component_surrogate_hsi_codesign
from optiresearch.schemas.component_surrogate_psf import ComponentSurrogateHSICoDesignSpec


def test_runtime_handler_runs_fresnel_codesign(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = run_component_surrogate_hsi_codesign(
        ComponentSurrogateHSICoDesignSpec(
            component_type="fresnel",
            steps=3,
            band_count=4,
            image_size=16,
            psf_size=9,
            batch_size=1,
        )
    )

    assert result.status == "succeeded"
    assert result.component_type == "fresnel"
    assert result.component_parameter_changed is True
    assert result.psf_requires_grad is True


def test_runtime_handler_runs_binary2phase_codesign(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = run_component_surrogate_hsi_codesign(
        ComponentSurrogateHSICoDesignSpec(
            component_type="binary2phase",
            steps=3,
            band_count=4,
            image_size=16,
            psf_size=9,
            batch_size=1,
        )
    )

    assert result.status == "succeeded"
    assert result.component_type == "binary2phase"
    assert result.component_parameter_changed is True
