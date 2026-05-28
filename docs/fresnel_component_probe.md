# Fresnel Component Probe

Phase 62 validates the **Fresnel** diffractive surface component from DeepLens
as a trainable, differentiable computational element.

## Component Details

- **Surface class:** `Fresnel`
- **Module:** `deeplens.diffractive_surface.fresnel`
- **Trainable parameter:** `f0` (focal length)
- **Backend ID:** `deeplens_fresnel_component`

## Usage

```bash
# Local probe
python -m optiresearch.cli run-deeplens-component-probe \
    --component fresnel --device cpu --max-steps 5

# Remote probe (WSL)
python -m optiresearch.cli run-remote-deeplens-component-probe \
    --worker-id windows_wsl --component fresnel --device cpu
```

## Success Criteria

| Field | Requirement |
|-------|------------|
| `status` | `succeeded` |
| `parameter_count` | `> 0` |
| `params_with_grad` | `> 0` |
| `gradient_norm` | `> 0` |
| `parameters_changed` | `true` |
| `loss_before` / `loss_after` | recorded |

## Evidence

- **Evidence level:** `diagnostic_evidence`
- **Claim ceiling:** `native_component_optimization` (if differentiable)
- **Does NOT support:** lens-level optimization, HSI improvement, real camera validation
