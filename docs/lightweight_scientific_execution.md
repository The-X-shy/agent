# Lightweight Scientific Execution

Phase 39 introduces `lightweight_scientific_execution` as a new evidence level alongside the existing `report_only`, `structured_unsupported`, and `local_execution_completed` levels.

## What It Is

A local, metric-producing experiment that uses:
- **Synthetic HSI data** (Gaussian spectral blobs)
- **FFT-based PSF generation** (Fraunhofer diffraction proxy)
- **No DeepLens dependency** — pure PyTorch
- **MSE-only objective** — no multi-objective loss
- **Simple linear reconstructor** — no CNN or complex model architecture

## What It Produces

Real, measured metrics:
- `reconstruction_loss_before` / `reconstruction_loss_after`
- `mse_before` / `mse_after`
- `psnr_before` / `psnr_after`
- `best_reconstruction_loss`
- `improvement_detected` (boolean)
- `metrics_valid` (boolean)
- `execution_time_sec`

## What It Cannot Claim

- Native DeepLens simulation
- Physical optical validation
- Real HSI performance
- Native GeoLens geometric PSF behavior
- Wave-optics / coherent propagation

## Claim Ceiling

`lightweight_scientific_execution` — claims are capped at synthetic metric experiments. The ClaimGate enforces:

| Claim Type | Allowed? |
|---|---|
| "Synthetic experiment shows MSE improvement" | Yes |
| "MSE-only objective completed successfully" | Yes |
| "Native DeepLens simulation validates..." | No — `lightweight_as_native_physical` |
| "Real HSI performance demonstrated..." | No — `synthetic_metric_as_real_hsi` |
| "Physical optical improvement..." | No — `lightweight_as_native_physical` |

## Position in Evidence Hierarchy

```
unsupported < mock_simulation < deeplens_integration_smoke <
native_component_optimization < native_hsi_proxy <
native_full_reconstruction_proxy < lightweight_scientific_execution <
native_lens_simulation < native_waveoptics <
stable_native_lens_hsi_codesign < rollback_protected_native_lens_hsi <
real_hsi_performance
```

Lightweight scientific execution sits between `native_full_reconstruction_proxy` and `native_lens_simulation` — it produces real metrics from optimization, but those metrics are on synthetic data with FFT proxy optics.

## Execution

```bash
python -m optiresearch.cli run-agent-plan-execution \
  --objective "recover from native GeoLens optical update instability" \
  --mode local \
  --execute-top-k 1
```

When the agent selects `objective_redesign_simpler_metric_mse_only` as the executable design, the lightweight scientific handler produces:
- `evidence_level=lightweight_scientific_execution`
- `scientific_execution=true` (derived from evidence level)
- Real metric values in `execution_result.metrics`
