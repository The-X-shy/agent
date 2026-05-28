# Binary2Phase Component Probe

Phase 62 validates the **Binary2Phase** phase surface component from DeepLens
as a trainable, differentiable computational element.

## Component Details

- **Surface class:** `Binary2Phase`
- **Module:** `deeplens.phase_surface.binary2`
- **Trainable parameters:** `d`, `order2`, `order4`, `order6`, `order8`, `order10`, `order12`
- **Backend ID:** `deeplens_binary2phase_component`

## Usage

```bash
# Local probe
python -m optiresearch.cli run-deeplens-component-probe \
    --component binary2phase --device cpu --max-steps 5

# Remote probe (WSL)
python -m optiresearch.cli run-remote-deeplens-component-probe \
    --worker-id windows_wsl --component binary2phase --device cpu
```

## Success Criteria

| Field | Requirement |
|-------|------------|
| `status` | `succeeded` |
| `parameter_count` | `>= 7` |
| `params_with_grad` | `> 0` |
| `gradient_norm` | `> 0` |
| `parameters_changed` | `true` |

## Evidence

- **Evidence level:** `diagnostic_evidence`
- **Claim ceiling:** `native_component_optimization` (if at least one parameter changes)
- **Does NOT support:** lens-level optimization, HSI improvement
