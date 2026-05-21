"""Tests for validate_autograd_chain helper."""

import torch
from optiresearch.adapters.deeplens_differentiable_bridge import validate_autograd_chain


def test_autograd_chain_detects_valid_graph():
    opt_param = torch.nn.Parameter(torch.tensor([1.0]))
    recon_param = torch.nn.Parameter(torch.tensor([2.0]))

    x = opt_param * 3.0
    y = x * recon_param
    loss = y.pow(2).mean()
    loss.backward()

    result = validate_autograd_chain(loss, [opt_param], [recon_param])
    assert result["autograd_graph_exists"] is True
    assert result["optical_gradient_norm"] > 0
    assert result["recon_gradient_norm"] > 0


def test_autograd_chain_no_grad_loss():
    opt_param = torch.nn.Parameter(torch.tensor([1.0]))
    loss = torch.tensor(5.0)

    result = validate_autograd_chain(loss, [opt_param])
    assert result["autograd_graph_exists"] is False
    assert result["failed_reason"] == "loss does not require grad"


def test_autograd_chain_zero_optical_grad():
    opt_param = torch.nn.Parameter(torch.tensor([1.0]))
    recon_param = torch.nn.Parameter(torch.tensor([2.0]))
    # Loss only depends on recon_param
    loss = recon_param.pow(2).mean()
    loss.backward()

    result = validate_autograd_chain(loss, [opt_param], [recon_param])
    assert result["optical_gradient_norm"] == 0.0
    assert result["recon_gradient_norm"] > 0


def test_autograd_chain_no_recon_params():
    opt_param = torch.nn.Parameter(torch.tensor([1.0]))
    loss = opt_param.pow(2).mean()
    loss.backward()

    result = validate_autograd_chain(loss, [opt_param])
    assert result["autograd_graph_exists"] is True
    assert result["recon_gradient_norm"] == 0.0
