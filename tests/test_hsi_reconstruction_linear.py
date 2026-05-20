import json

import numpy as np

from optiresearch.hsi.reconstruction import LinearSpectralReconstructor, run_linear_reconstruction


def test_linear_spectral_reconstructor_fits_and_predicts(tmp_path):
    measurements = np.ones((3, 1, 8, 8), dtype=np.float32)
    targets = np.stack([measurements[:, 0] * 0.2, measurements[:, 0] * 0.5, measurements[:, 0] * 0.8], axis=1)
    reconstructor = LinearSpectralReconstructor(output_bands=3)

    reconstructor.fit(measurements, targets)
    prediction = reconstructor.predict(measurements[:1])
    result = run_linear_reconstruction(measurements, targets, measurements[:1], targets[:1], [0], tmp_path)

    assert prediction.shape == (1, 3, 8, 8)
    assert prediction[0, 2].mean() > prediction[0, 0].mean()
    assert (tmp_path / "reconstruction_metrics.json").exists()
    assert (tmp_path / "reconstructed_test.npz").exists()
    metrics = json.loads((tmp_path / "reconstruction_metrics.json").read_text(encoding="utf-8"))
    assert "PSNR" in metrics
    assert result["metrics"]["network_type"] == "linear_baseline"
