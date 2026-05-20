from optiresearch.schemas.hsi import (
    build_default_hsi_forward_model_spec,
    build_default_hsi_reconstruction_spec,
    build_default_synthetic_hsi_dataset_spec,
)


def test_hsi_default_schemas_are_valid():
    dataset = build_default_synthetic_hsi_dataset_spec()
    forward = build_default_hsi_forward_model_spec(optical_artifact_id="artifact_psf", psf_cube_uri="runs/psf_cube.npz")
    reconstruction = build_default_hsi_reconstruction_spec(output_bands=dataset.spectral_bands)

    assert dataset.schema_version == "0.1-draft"
    assert dataset.source == "synthetic"
    assert dataset.spectral_bands == 31
    assert forward.wavelength_bands == 31
    assert forward.measurement_type == "single_shot"
    assert reconstruction.network_type in {"linear_baseline", "optical_conditioned_linear"}
    assert reconstruction.output_bands == 31
