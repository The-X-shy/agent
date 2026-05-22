"""Tests for autograd graph auditor."""

import torch
from optiresearch.diagnostics.autograd_auditor import (
    audit_autograd_graph,
    compare_gradient_strength,
    detect_detach,
    inspect_tensor_requires_grad,
    summarize_gradient_flow,
)


def test_audit_clean_graph():
    x = torch.tensor([1.0], requires_grad=True)
    y = x * 2.0
    loss = (y - 3.0) ** 2
    loss.backward()
    report = audit_autograd_graph(loss, {"x": x})
    assert report.verdict == "clean"
    assert report.loss_requires_grad is True
    assert report.parameters_with_grad == 1
    assert len(report.suspected_breaks) == 0


def test_audit_detached_tensor():
    x = torch.tensor([1.0], requires_grad=True)
    y = x.detach() * 2.0
    loss = (y - 3.0) ** 2
    # loss does not require grad because y was detached
    # So backward would fail — just audit without calling backward
    report = audit_autograd_graph(loss, {"x": x})
    assert report.loss_requires_grad is False
    assert report.verdict == "broken"


def test_audit_loss_not_requiring_grad():
    x = torch.tensor([1.0], requires_grad=True)
    y = x.detach() * 2.0
    loss = (y - 3.0) ** 2
    report = audit_autograd_graph(loss, {"x": x})
    assert report.loss_requires_grad is False
    assert report.verdict == "broken"


def test_audit_rollback_does_not_false_flag():
    """Rollback parameters_changed=False should NOT flag as broken."""
    x = torch.tensor([1.0], requires_grad=True)
    y = x * 2.0
    loss = (y - 3.0) ** 2
    loss.backward()
    report = audit_autograd_graph(loss, {"x": x}, rollback_parameters_changed=False)
    assert report.verdict == "clean"
    assert report.rollback_parameters_changed is False


def test_audit_zero_grad_parameter():
    x = torch.tensor([1.0], requires_grad=True)
    y = x * 0.0  # gradient will be 0
    loss = (y - 0.0) ** 2
    loss.backward()
    report = audit_autograd_graph(loss, {"x": x})
    assert report.verdict == "clean"
    assert "x" in report.zero_grad_parameters


def test_audit_missing_grad_parameter():
    x = torch.tensor([1.0], requires_grad=True)
    loss = x  # No backward called
    report = audit_autograd_graph(loss, {"x": x})
    assert len(report.missing_grad_parameters) == 1


def test_inspect_tensor_requires_grad():
    tensors = {
        "a": torch.tensor([1.0], requires_grad=True),
        "b": torch.tensor([2.0], requires_grad=False),
    }
    result = inspect_tensor_requires_grad(tensors)
    assert result["a"] is True
    assert result["b"] is False


def test_detect_detach_normal_tensor():
    x = torch.tensor([1.0], requires_grad=True)
    result = detect_detach(x, "x")
    # Not detached — should be None
    assert result is None


def test_detect_detach_actually_detached():
    x = torch.tensor([1.0], requires_grad=True)
    y = x.detach()
    result = detect_detach(y, "y")
    assert result is None  # requires_grad is False, so not flagged


def test_summarize_gradient_flow():
    x = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    loss = (x**2).sum()
    loss.backward()
    info = summarize_gradient_flow([x])
    assert info["parameter_count"] == 1
    assert info["parameters_with_grad"] == 1
    assert len(info["missing_grad_parameters"]) == 0


def test_compare_gradient_strength():
    a = torch.tensor([1.0], requires_grad=True)
    b = torch.tensor([3.0], requires_grad=True)
    loss = a * 2 + b * 4
    loss.backward()
    rel = compare_gradient_strength({"a": a.grad, "b": b.grad})
    assert "a" in rel
    assert "b" in rel
    assert abs(rel["a"] - 2.0 / 6.0) < 1e-4


def test_audit_with_module():
    class SimpleModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 2)

        def forward(self, x):
            return self.linear(x)

    m = SimpleModule()
    x = torch.rand(2, 4)
    y = m(x)
    loss = y.sum()
    loss.backward()
    params = {n: p for n, p in m.named_parameters()}
    report = audit_autograd_graph(loss, params, module=m)
    assert report.verdict == "clean"
    assert report.parameters_with_grad == 2


def test_audit_multiple_params():
    a = torch.tensor([1.0, 2.0], requires_grad=True)
    b = torch.tensor([3.0, 4.0], requires_grad=True)
    loss = (a * b).sum()
    loss.backward()
    report = audit_autograd_graph(loss, {"a": a, "b": b})
    assert report.parameter_count == 2
    assert report.parameters_with_grad == 2
    assert report.verdict == "clean"
