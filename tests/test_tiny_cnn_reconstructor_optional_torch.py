import numpy as np
import pytest

from optiresearch.hsi.reconstruction import TinyCNNReconstructor, torch_available


def test_tiny_cnn_reports_torch_availability_without_hard_dependency():
    rec = TinyCNNReconstructor(output_bands=3, epochs=1)

    assert rec.available() == torch_available()
    if not rec.available():
        assert rec.error["code"] == "TORCH_NOT_AVAILABLE"


def test_tiny_cnn_training_small_if_torch_available(tmp_path):
    if not torch_available():
        pytest.skip("torch unavailable")

    x = np.random.default_rng(0).random((2, 1, 8, 8), dtype=np.float32)
    y = np.repeat(x, 3, axis=1)
    rec = TinyCNNReconstructor(output_bands=3, hidden_channels=4, depth=2, epochs=1, batch_size=1)

    result = rec.run(x, y, x[:1], y[:1], [0], tmp_path)

    assert result.status == "succeeded"
    assert result.metrics["network_type"] == "tiny_cnn"
    assert result.artifact_paths

