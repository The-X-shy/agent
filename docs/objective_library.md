# Objective Library

The objective library provides composable, differentiable loss functions for
optical and HSI optimization.

## Loss Functions

### Optical

| Function | Purpose |
|----------|---------|
| psf_width_loss | Penalises deviation of PSF spatial width from target |
| psf_centroid_loss | Penalises PSF centroid drift |
| psf_energy_loss | Penalises deviation of total PSF energy |
| psf_smoothness_loss | Total-variation smoothness regularizer |
| spot_size_loss | Penalises pixels above fractional intensity threshold |
| field_consistency_loss | Penalises PSF shape variation across field |

### HSI

| Function | Purpose |
|----------|---------|
| reconstruction_mse | Mean squared error between reconstructed and target HSI |
| spectral_angle_loss | Spectral angle mapper — penalises spectral shape mismatch |
| measurement_consistency_loss | Penalises inconsistency between forward and measurement |
| spectral_smoothness_loss | Penalises high-frequency variation along spectral dim |
| band_weighted_mse | Band-weighted MSE |
| task_aligned_hsi_loss | Combined task-aligned HSI loss |

### Regularizers

| Function | Purpose |
|----------|---------|
| optical_param_l2 | L2 penalty on optical parameters |
| optical_param_delta_limit | Penalises parameter changes exceeding max_delta |
| psf_energy_preservation | Encourages PSF energy conservation |
| psf_centroid_preservation | Penalises PSF centroid drift |
| psf_width_preservation | Discourages PSF width deviation |
| rollback_penalty | Soft penalty that grows with rollback count |

## Preset Profiles

### stable_lens_hsi_codesign

Phase 23 recommended stable training config:
- Losses: reconstruction_mse (1.0), spectral_angle_loss (0.05), psf_energy_preservation (0.1)
- Compatible: deeplens_geolens_geometric
- Claim: stable_native_lens_hsi_codesign

### psf_quality_probe

Basic PSF quality evaluation:
- Losses: psf_width_loss (1.0), psf_centroid_loss (0.5), psf_energy_loss (0.5)
- Compatible: mock_deeplens, deeplens_blackbox_source_psf, deeplens_geolens_geometric

### component_optimization

Component-level optical surface optimization:
- Losses: psf_width_loss (1.0), psf_energy_loss (0.5), optical_param_l2 (1e-3)
- Compatible: deeplens_fresnel_component, deeplens_binary2phase_component

## CLI

```bash
python -m optiresearch.cli list-objective-profiles
python -m optiresearch.cli inspect-objective-profile --profile-id stable_lens_hsi_codesign
```

## Programmatic API

```python
from optiresearch.objectives.optical_objectives import (
    list_objective_profiles, get_objective_profile,
)
from optiresearch.objectives.hsi_objectives import reconstruction_mse
from optiresearch.objectives.regularizers import psf_energy_preservation

# Use preset profile
profile = get_objective_profile("stable_lens_hsi_codesign")

# Compose custom loss
loss = 1.0 * reconstruction_mse(recon, target) + 0.1 * psf_energy_preservation(psf)
```
