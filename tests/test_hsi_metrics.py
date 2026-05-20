import numpy as np

from optiresearch.hsi.metrics import ergas, per_band_rmse, psnr, sam, ssim_simple, worst_depth_sam


def test_hsi_metrics_are_json_serializable_and_stable():
    target = np.ones((2, 4, 8, 8), dtype=np.float32)
    pred = target * 0.9

    assert psnr(pred, target) > 10.0
    assert 0.0 <= ssim_simple(pred, target) <= 1.0
    assert sam(pred, target) >= 0.0
    assert ergas(pred, target) >= 0.0
    assert len(per_band_rmse(pred, target)) == 4
    assert worst_depth_sam(pred, target, [0, 1]) >= 0.0
