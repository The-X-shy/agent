import numpy as np
import pytest

from optiresearch.hsi.reconstruction import UNetTinyReconstructor, run_reconstruction, torch_available


def test_unet_tiny_structured_result_without_default_torch_requirement(tmp_path):
    x = np.ones((2, 1, 8, 8), dtype=np.float32)
    y = np.ones((2, 3, 8, 8), dtype=np.float32)
    rec = UNetTinyReconstructor(output_bands=3, hidden_channels=4, epochs=1, batch_size=1)

    result = rec.run(x, y, x[:1], y[:1], [0], tmp_path)

    assert result.status in {"succeeded", "skipped"}
    if not torch_available():
        assert result.error_code == "TORCH_NOT_AVAILABLE"


def test_run_reconstruction_accepts_unet_tiny_when_torch_available(tmp_path):
    if not torch_available():
        pytest.skip("torch unavailable")

    x = np.ones((2, 1, 8, 8), dtype=np.float32)
    y = np.ones((2, 3, 8, 8), dtype=np.float32)
    result = run_reconstruction("unet_tiny", x, y, x[:1], y[:1], [0], tmp_path, train_options={"epochs": 1, "hidden_channels": 4})

    assert result["status"] == "succeeded"
    assert result["metrics"]["network_type"] == "unet_tiny"

