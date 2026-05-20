import json

import numpy as np

from optiresearch.hsi.reconstruction import ReconstructorResult, TinyCNNReconstructor


def test_tiny_cnn_reconstructor_returns_structured_result(tmp_path):
    x = np.ones((2, 1, 8, 8), dtype=np.float32)
    y = np.ones((2, 3, 8, 8), dtype=np.float32) * 0.5
    rec = TinyCNNReconstructor(output_bands=3, hidden_channels=4, depth=2, epochs=1, batch_size=1, seed=7)

    result = rec.run(x, y, x[:1], y[:1], [0], tmp_path)

    assert isinstance(result, ReconstructorResult)
    assert result.status in {"succeeded", "skipped"}
    if result.status == "skipped":
        assert result.error_code == "TORCH_NOT_AVAILABLE"
    else:
        assert "PSNR" in result.metrics
        assert (tmp_path / "checkpoint.pt").exists()
        assert json.loads((tmp_path / "reconstruction_manifest.json").read_text())["network_type"] == "tiny_cnn"

