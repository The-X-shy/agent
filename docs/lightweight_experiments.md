# Lightweight Experiments (Phase 28)

Pure-PyTorch differentiable optics experiments that run without DeepLens.

## Core Technique

FFT-based Fraunhofer diffraction PSF generation:

```
PSF = |FFT{exp(i * phase_mask)}|²
```

The phase mask is a learnable 2D parameter. Each wavelength band adds a phase offset. The FFT of the complex field gives the far-field PSF. This is standard Fourier optics, implemented entirely in PyTorch with zero DeepLens dependencies.

## Available Experiments

### 1. Lightweight PSF Probe

```python
from optiresearch.runtime.lightweight_experiments import run_lightweight_psf_probe
result = run_lightweight_psf_probe(backend_id="phase_to_fft_proxy")
```

- Generates PSFs and computes metrics (energy, FWHM, centroid)
- Completes in <5 seconds
- Returns `ControllerResult` with `evidence_level="deeplens_integration_smoke"`

### 2. Lightweight Stable Lens HSI

```python
result = run_lightweight_stable_lens_hsi(
    backend_id="phase_to_fft_proxy",
    max_steps=5,
    optical_lr=1e-6,
)
```

- Jointly optimizes phase mask (optical) + linear reconstructor (digital)
- Rollback protection: reverts phase mask if loss increases
- Completes in <60 seconds on CPU
- Returns metrics: loss_before/after, rollback_count, accepted_updates, gradient norms

### 3. Lightweight Ablation

```python
result = run_lightweight_ablation(
    backend_id="phase_to_fft_proxy",
    max_configs=2,
    max_steps=3,
)
```

- Compares different optical LRs (1e-6 vs 1e-5)
- Reports winner and per-config metrics

## Controller Integration

The `lightweight_psf_probe` task type is registered in `ExperimentControllerV2`:
- `_TASK_REQUIRED_CEILING`: `deeplens_integration_smoke` (lowest ceiling)
- `_dispatch_local` routes to `_run_lightweight_psf_probe()`

## Claim Ceiling

Lightweight experiments operate under `native_full_reconstruction_proxy` or `deeplens_integration_smoke` claim ceilings — intentionally lower than DeepLens-backed experiments. The claim gate enforces these boundaries.
