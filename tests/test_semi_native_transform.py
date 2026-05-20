import numpy as np

from optiresearch.adapters.deeplens_encoder_strategies import get_deeplens_encoder_strategy
from optiresearch.adapters.deeplens_semi_native import SemiNativeTransform
from optiresearch.schemas.experiment import build_default_mock_edof_hsi_experiment


def test_semi_native_transform_supports_conventional_with_paraxial():
    transform = SemiNativeTransform()
    spec = build_default_mock_edof_hsi_experiment("semi native", encoder_type="conventional")
    strategy = get_deeplens_encoder_strategy("conventional")
    capabilities = {"paraxial_lens_available": True}
    api_probe = {"candidate_phase_or_doe_classes": [], "candidate_surface_classes": []}

    assert transform.supports("conventional", api_probe, capabilities) is True
    config = transform.build_config(spec, strategy)
    before = transform.apply_before_psf(config)
    cube, manifest = transform.apply_after_psf_if_needed(np.ones((2, 2, 4, 4), dtype=np.float32), before)

    assert cube.shape == (2, 2, 4, 4)
    assert manifest["selected_realization_level"] == "semi_native"
    assert manifest["semi_native_succeeded"] is True


def test_semi_native_transform_does_not_upgrade_postprocess_only_encoder():
    transform = SemiNativeTransform()
    api_probe = {"candidate_phase_or_doe_classes": [], "candidate_surface_classes": []}

    assert transform.supports("edof", api_probe, {"paraxial_lens_available": True}) is False
