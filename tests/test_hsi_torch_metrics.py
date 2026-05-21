"""Tests for torch HSI metrics."""

import torch
from optiresearch.hsi.torch_metrics import torch_mse, torch_psnr, torch_sam, torch_ergas_proxy


def test_torch_mse_scalar():
    a = torch.randn(2, 4, 16, 16)
    b = torch.randn(2, 4, 16, 16)
    mse = torch_mse(a, b)
    assert mse.dim() == 0
    assert mse.item() > 0


def test_torch_psnr():
    a = torch.ones(2, 4, 16, 16)
    b = torch.ones(2, 4, 16, 16) * 0.9
    psnr = torch_psnr(a, b)
    assert psnr.dim() == 0
    assert psnr.item() > 0


def test_torch_sam():
    a = torch.ones(2, 4, 16, 16)
    b = torch.ones(2, 4, 16, 16)
    sam = torch_sam(a, b)
    assert sam.item() < 1e-4


def test_torch_ergas():
    a = torch.ones(2, 4, 16, 16)
    b = torch.ones(2, 4, 16, 16)
    ergas = torch_ergas_proxy(a, b)
    assert ergas.item() < 1e-4


def test_metrics_are_pure_torch():
    import optiresearch.hsi.torch_metrics as mod
    source = open(mod.__file__).read()
    assert "import numpy" not in source
    assert "from numpy" not in source
