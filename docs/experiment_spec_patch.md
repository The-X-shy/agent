# Experiment Spec Patch

## Overview

The `experiment_spec_patch` field allows the LLM planner to override default
experiment parameters in the `ExperimentSpecV2.spec_payload`. Patches are
validated for safety before being applied.

## Allowed Overrides

| Field | Description |
|-------|-------------|
| `optical_lr` | Learning rate for optical parameters |
| `recon_lr` | Learning rate for reconstructor |
| `max_steps` | Maximum optimization steps |
| `rollback_on_loss_increase` | Enable/disable rollback protection |
| `objective_profile` | Objective profile name |
| `lightweight_mode` | Force lightweight experiment path |
| `device` | Compute device (cpu/cuda) |
| `bands` | Number of spectral bands |
| `image_size` | HSI image size |
| `psf_size` | PSF kernel size |

## Blocked Overrides

These fields are stripped from patches for safety:

| Field | Reason |
|-------|--------|
| `backend_id` | Backend must match loop configuration |
| `task_type` | Task type determined by strategy compiler |
| `execution_target` | Must stay `local` unless explicitly allowed |
| `claim_ceiling` | Determined by backend, cannot be overridden |
| `shell_command` | No arbitrary shell execution |
| `file_path` | No arbitrary file access |

## How Patches Are Applied

1. LLM proposes `experiment_spec_patch` in its planner response.
2. `compile_experiment_spec()` receives the patch.
3. `_build_payload()` builds default payload for the action.
4. Safety filter strips disallowed keys from the patch.
5. Safe patch is merged over defaults via `payload.update(safe_patch)`.

## Example

LLM proposal:
```json
{
  "experiment_spec_patch": {
    "optical_lr": 1e-7,
    "max_steps": 8,
    "backend_id": "deeplens_geolens_geometric"
  }
}
```

After safety filter:
```python
safe_patch = {"optical_lr": 1e-7, "max_steps": 8}
# backend_id stripped — backend is determined by loop config
```
