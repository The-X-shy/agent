# Objective Redesign Handler

The `objective_redesign_simpler_metric_mse_only` handler is the first dedicated scientific handler in the Agent Plan Execution Loop.

## Design

When the `EvidenceStrategyReasoner` generates the `objective_redesign_simpler_metric` strategy, the `ExperimentDesignGenerator` creates a design with:
- `design_id`: `objective_redesign_simpler_metric_mse_only`
- `backend_id`: `deeplens_geolens_geometric` (aspirational)
- `task_type`: `stable_lens_hsi_codesign`
- `spec_payload.loss_weights`: `{"mse": 1.0, "spectral_angle": 0.0, "measurement_consistency": 0.0}`

## Execution

The handler in `agent_plan_execution_loop.py` recognizes the design via `_is_lightweight_scientific_design()` and routes it to `_execute_lightweight_scientific_design()`, which calls `run_lightweight_mse_only_hsi()` from `lightweight_experiments.py`.

The experiment:
1. Generates synthetic HSI target (Gaussian blobs, 4 bands, 16x16)
2. Generates FFT-based PSF from learnable phase mask (4 bands, 15x15)
3. Jointly optimizes phase mask + linear reconstructor with MSE loss
4. Tracks loss before/after, best loss, gradient norms, improvement
5. Returns `ControllerResult` with `evidence_level=lightweight_scientific_execution`

## Skill Registration

The handler is also registered as a skill:
- **Skill ID**: `lightweight_scientific_hsi_mse_only`
- **Registry**: `SkillRegistryV2._register_builtins()`
- **Runtime**: `SkillRuntimeV2._dispatch_lightweight_scientific_hsi()`

## Metrics Produced

| Metric | Type | Description |
|---|---|---|
| `reconstruction_loss_before` | float | MSE at step 0 |
| `reconstruction_loss_after` | float | MSE at final step |
| `best_reconstruction_loss` | float | Best MSE across all steps |
| `mse_before` | float | Same as loss_before (MSE-only) |
| `mse_after` | float | Same as loss_after (MSE-only) |
| `psnr_before` | float | PSNR at step 0 |
| `psnr_after` | float | PSNR at final step |
| `improvement_detected` | bool | True if loss decreased |
| `metrics_valid` | bool | Always True on success |
| `execution_time_sec` | float | Wall-clock time |

## Metadata

| Field | Value |
|---|---|
| `synthetic_data` | True |
| `physical_backend` | False |
| `mse_only_objective` | True |
| `deepens_used` | False |
| `psf_generation_method` | fft_fraunhofer |

## Caveats

- Uses synthetic HSI data — no real sensor noise or calibration effects
- Uses FFT Fraunhofer proxy — not native DeepLens geometric ray-tracing
- MSE-only objective — does not test multi-objective loss stability
- Linear reconstructor — does not test CNN or transformer-based reconstruction
