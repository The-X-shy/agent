import numpy as np

from optiresearch.hsi.forward_model import HSIForwardModel, align_psf_to_hsi_wavelengths, validate_psf_hsi_compatibility
from optiresearch.schemas.hsi import build_default_hsi_forward_model_spec


def test_forward_model_loads_raw_base_psf_cube_key(tmp_path):
    cube = np.ones((2, 3, 5, 5), dtype=np.float32)
    path = tmp_path / "raw_base_psf_cube.npz"
    np.savez_compressed(path, raw_base_psf_cube=cube)

    loaded = HSIForwardModel(build_default_hsi_forward_model_spec(wavelength_bands=3)).load_psf_cube(str(path))

    assert loaded.shape == (2, 3, 5, 5)


def test_validate_psf_hsi_compatibility_reports_mismatch_without_resample():
    result = validate_psf_hsi_compatibility(np.ones((2, 3, 5, 5)), psf_wavelengths=[450, 550, 650], hsi_wavelengths=[450, 500, 550, 600])

    assert result["status"] == "error"
    assert result["error_code"] == "BAND_MISMATCH"


def test_align_psf_to_hsi_wavelengths_interpolates_band_axis():
    psf = np.stack([np.ones((1, 4, 4)) * value for value in [1.0, 2.0, 3.0]], axis=1)

    aligned = align_psf_to_hsi_wavelengths(psf, [450.0, 550.0, 650.0], [450.0, 500.0, 550.0, 600.0, 650.0])

    assert aligned.shape == (1, 5, 4, 4)
    assert float(aligned[0, 1, 0, 0]) == 1.5

