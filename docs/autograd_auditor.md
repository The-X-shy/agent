# Autograd Auditor

The autograd auditor analyzes gradient flow through optical parameters to
detect broken differentiable links.

## Capabilities

- `inspect_tensor_requires_grad` — check which tensors track gradients
- `trace_loss_to_parameters` — verify loss can reach parameters
- `detect_detach` — find detached non-leaf tensors in the graph
- `detect_numpy_conversion_risk` — scan for numpy conversions
- `detect_no_grad_region` — detect no_grad contexts
- `summarize_gradient_flow` — gradient norm statistics
- `compare_gradient_strength` — relative gradient magnitudes

## Verdict Levels

- **clean**: Gradient flow is intact
- **suspicious**: Potential issues detected (e.g., some zero gradients)
- **broken**: Clear autograd break (e.g., loss detached from graph)

## Rollback Awareness

The auditor distinguishes between:
- **Normal zero-grad:** `rollback_parameters_changed=False` → `verdict: clean`
  (parameters were restored from snapshot — zero change is expected)
- **True autograd break:** `loss.requires_grad=False` → `verdict: broken`
  (graph is detached — no gradient can flow)

## CLI

```bash
python -m optiresearch.cli audit-autograd-graph
```

## Programmatic API

```python
import torch
from optiresearch.diagnostics.autograd_auditor import audit_autograd_graph

x = torch.tensor([1.0], requires_grad=True)
y = x * 2.0
loss = (y - 3.0) ** 2
loss.backward()

report = audit_autograd_graph(loss, {"x": x})
print(report.verdict)       # "clean"
print(report.gradient_norms) # {"x": 4.0}
```
