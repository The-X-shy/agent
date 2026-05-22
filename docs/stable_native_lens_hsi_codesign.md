# Stable Native Lens HSI Co-Design (Phase 23)

## Overview

Phase 23 introduced a stable training loop for native differentiable lens HSI co-design.
The key innovation is a rollback mechanism that protects against harmful optical updates.

## Key Results

- **Local small_lr**: reconstruction_loss 1.59 → 1.10
- **Remote GeoLens geometric**: reconstruction_loss 0.0883 → 0.0887
  - accepted_update_count=0, rejected_update_count=7, rollback_count=7
  - optical_parameters_changed=false, stable_training_succeeded=false

## Architecture

Three-phase training:
1. **Reconstructor Warmup** (3 steps): Optics frozen, reconstructor-only training
2. **Joint Finetune**: Both optics and reconstructor train, with rollback on loss increase
3. **Final Reconstructor Adaptation**: Optics frozen again, 2 final reconstructor steps

## Rollback Mechanism

Before each optimizer step, optical parameters are snapshotted. After the step,
if loss increases beyond a tolerance, parameters are restored from the snapshot.

## Key Findings

1. Rollback mechanism is effective at preventing loss explosion
2. GeoLens geometric PSF gradients are large (~1737) and require small optical_lr (1e-6)
3. Remote results show 0 accepted optical updates — environment differences matter
4. Rollback protects but does not prove optical improvement

## Phase 24 Integration

Phase 24 wraps this in `ExperimentControllerV2` with:
- Backend registry: `deeplens_geolens_geometric` with claim ceiling `native_lens_simulation`
- Objective profile: `stable_lens_hsi_codesign`
- Strategy engine: recommends small_lr + rollback when gradients are large
- Claim gate: prevents claiming rollback protection as optical improvement
