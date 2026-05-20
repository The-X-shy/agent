import numpy as np

from optiresearch.hsi.forward_model import HSIForwardModel
from optiresearch.schemas.hsi import build_default_hsi_forward_model_spec


def test_hsi_forward_model_renders_single_channel_measurement(tmp_path):
    psf_cube = np.ones((2, 4, 3, 3), dtype=np.float32)
    psf_cube = psf_cube / psf_cube.sum(axis=(-1, -2), keepdims=True)
    np.savez_compressed(tmp_path / "psf_cube.npz", psf_cube=psf_cube)
    hsi = np.ones((4, 8, 8), dtype=np.float32)
    model = HSIForwardModel(build_default_hsi_forward_model_spec(psf_cube_uri=str(tmp_path / "psf_cube.npz"), depth_planes=2, wavelength_bands=4))

    loaded = model.load_psf_cube(str(tmp_path / "psf_cube.npz"))
    measurement = model.render_measurement(hsi, loaded, depth_index=1)
    batch = model.render_batch(np.stack([hsi, hsi * 0.5]), loaded, depth_indices=[0, 1])

    assert loaded.shape == (2, 4, 3, 3)
    assert measurement.shape == (1, 8, 8)
    assert batch["measurements"].shape == (2, 1, 8, 8)
    assert batch["targets"].shape == (2, 4, 8, 8)
