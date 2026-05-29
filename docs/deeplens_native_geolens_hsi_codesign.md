# DeepLens Native GeoLens HSI Co-Design

## Overview

The full native DeepLens GeoLens geometric HSI co-design path uses actual
`deeplens.geolens` ray-tracing PSF generation for HSI reconstruction co-design,
as opposed to the FFT-based proxy path.

GeoLens trainable parameters must be discovered through DeepLens
`get_optimizer_params()` / `get_optimizer()`. `GeoLens` is not a standard
`torch.nn.Module`, so `geolens.parameters()` is not a valid trainability check.
The geometric PSF path uses float32 tensors.

## Entry Point

`run_stable_native_lens_hsi_codesign(spec: StableNativeLensHSISpec)` in
`optiresearch/runtime/stable_native_lens_hsi_loop.py`

## Three-Phase Training

1. **Reconstructor Warmup**: PSF is detached (optics frozen), reconstructor trained alone
2. **Joint Finetune**: PSF regenerated each step, both optics and reconstructor optimized
3. **Final Adaptation**: Optics frozen, final reconstructor refinement

## Stability Mechanisms

- Gradient clipping (optical: 1.0, reconstructor: 5.0)
- Rollback on loss increase
- PSF energy/width regularization
- Optical update interval (decouples optical/reconstructor update frequency)
- Native autograd audit fields:
  `trainable_param_count`, `params_with_grad`, `graph_connected`,
  `psf_requires_grad`, `loss_requires_grad`

## Execution Fidelity

| Fidelity | PSF Source | Evidence Level |
|---|---|---|
| `deeplens_native_geometric` | `deeplens.geolens.psf(model="geometric")` | `native_lens_simulation` |
| `lightweight_proxy` | `torch.fft.fft2(exp(i*phase))` | `native_full_reconstruction_proxy` |

## CLI

```bash
python -m optiresearch.cli run-deeplens-native-geolens-hsi-codesign \
  --lens-file auto:cooke --dataset synthetic \
  --reconstructor differentiable_linear --max-steps 5 \
  --optical-lr 1e-6 --rollback-on-loss-increase

# Remote on WSL
python -m optiresearch.cli run-remote-stable-native-lens-hsi-codesign \
  --worker-id windows_wsl --candidate GeoLensCooke \
  --reconstructor differentiable_linear --max-steps 5
```

## Platform Compatibility

- **WSL/Linux with DeepLens**: Full native path works
- **macOS**: GeoLens API IndexError caught early, returns structured `unsupported` with `GEOLENS_PSF_GEOMETRIC_FAILED_INDEXERROR`
- **No DeepLens**: Returns `BUILD_FAILED` error code
