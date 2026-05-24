# Param Reduction Sweep Handler

Phase 40 adds `param_reduction_sweep` as the second locally executable scientific handler, alongside `objective_redesign_simpler_metric_mse_only`.

## What It Does

Sweeps low-dimensional pseudo-optical parameter vectors:
- **k=1**: Single scalar parameter controlling PSF width
- **k=2**: Two parameters controlling PSF width and centroid
- **k=3**: Three parameters controlling PSF width, centroid, and spectral weighting

Each configuration runs 2-5 optimization steps on synthetic HSI data using FFT-based PSF generation (no DeepLens). The best k is selected based on lowest reconstruction loss.

## Metrics Produced

| Metric | Description |
|---|---|
| `configs_tested` | Number of k values swept |
| `best_k` | k value with lowest loss |
| `reconstruction_loss_before` | MSE at step 0 for best config |
| `reconstruction_loss_after` | MSE at final step for best config |
| `best_reconstruction_loss` | Lowest MSE across all steps |
| `mse_before/after` | Same as loss (MSE-only) |
| `psnr_before/after` | PSNR for best config |
| `improvement_detected` | True if loss decreased |
| `metrics_valid` | Always True on success |
| `accepted_update_count` | Steps where loss improved |

## Evidence Level

`lightweight_scientific_execution` — same ceiling as the MSE-only handler.

## Caveats

- Synthetic HSI data only — not real sensor measurements
- Low-dimensional pseudo-optical parameters — not native lens optimization
- FFT Fraunhofer PSF proxy — not native DeepLens geometric ray-tracing
- Does not test multi-objective loss stability

## Skill

- **Skill ID**: `param_reduction_sweep`
- **Runtime**: `SkillRuntimeV2._dispatch_param_reduction_sweep()`
- **Underlying**: `run_param_reduction_sweep_lightweight()` in `local_scientific_handlers.py`
