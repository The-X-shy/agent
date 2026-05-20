"""Test TinyCNNReconstructor optional backend."""

import pytest

from optiresearch.hsi.reconstruction import TinyCNNReconstructor


def test_tinycnn_initialization():
    rec = TinyCNNReconstructor(output_bands=3)
    if rec.error is not None:
        assert rec.error["code"] == "TORCH_NOT_AVAILABLE"
    else:
        assert rec._has_torch


def test_tinycnn_fit_without_torch():
    try:
        import torch  # noqa: F401
        pytest.skip("torch is available")
    except ImportError:
        pass
    rec = TinyCNNReconstructor(output_bands=3)
    assert rec.error is not None
    assert rec.error["code"] == "TORCH_NOT_AVAILABLE"
    # fit should be a no-op
    import numpy as np
    x = np.ones((4, 1, 8, 8), dtype=np.float32)
    y = np.ones((4, 3, 8, 8), dtype=np.float32)
    rec.fit(x, y)
    assert not rec._trained
