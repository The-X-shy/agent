# Remote GeoLens Diagnostics

Remote diagnostic execution for DeepLens GeoLens gradient analysis and trainable parameter inspection.

## Available Diagnostics

| Diagnostic | CLI Command | Purpose |
|---|---|---|
| Trainable Parameter Inspection | `run-remote-deeplens-trainable-parameter-inspection` | Enumerate GeoLens parameters, classify by gradient behavior |
| Autograd Audit | `run-remote-deeplens-autograd-audit` | Verify autograd chain from PSF through loss |
| Curriculum Probe | `run-remote-deeplens-curriculum-probe` | Staged difficulty progression (synthetic data) |
| Regularized Probe | `run-remote-deeplens-regularized-probe` | PSF energy/centroid regularization (synthetic data) |

## Usage

```bash
# Trainable parameter inspection
python -m optiresearch.cli run-remote-deeplens-trainable-parameter-inspection \
  --worker-id windows_wsl \
  --lens-file auto:cooke \
  --device cpu

# Autograd audit
python -m optiresearch.cli run-remote-deeplens-autograd-audit \
  --worker-id windows_wsl \
  --lens-file auto:cooke \
  --device cpu
```

## Output Metrics

### Trainable Parameter Inspection
- `trainable_param_count`, `parameter_count`
- `params_with_grad`, `zero_gradient_parameters`
- `grad_norm_max`, `grad_norm_mean`
- `resolved_lens_file`, `lens_resolution_source`

### Autograd Audit
- `graph_connected`, `psf_requires_grad`, `loss_requires_grad`
- `detach_suspected`, `candidate_update_changes_parameter`
- `grad_norm_max`, `recommended_next_strategy`

## Report

```bash
python -m optiresearch.cli export-remote-diagnostic-report --remote-job-id <id>
```

## Constraints

- Evidence level: `diagnostic_evidence` only
- No optical improvement claims from diagnostic data
- Remote commands validated through strict allowlist
