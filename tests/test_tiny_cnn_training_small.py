import numpy as np
import pytest

from optiresearch.hsi.reconstruction import TinyCNNReconstructor, torch_available


@pytest.mark.skipif(not torch_available(), reason="torch unavailable")
def test_tiny_cnn_trains_one_epoch_on_tiny_data(tmp_path):
    x = np.random.default_rng(11).random((2, 1, 8, 8), dtype=np.float32)
    y = np.repeat(x, 2, axis=1)

    result = TinyCNNReconstructor(output_bands=2, hidden_channels=4, depth=2, epochs=1, batch_size=1).run(x, y, x[:1], y[:1], [0], tmp_path)

    assert result.status == "succeeded"
    assert (tmp_path / "training_log.json").exists()
