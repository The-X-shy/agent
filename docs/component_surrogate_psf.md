# Component Surrogate PSF

`ComponentSurrogatePSF` turns validated component-level parameter sets into a
small differentiable PSF kernel for controlled HSI experiments.

## Supported Components

| Component | Parameters | Surrogate |
|---|---:|---|
| Fresnel | `f0` | Gaussian width with radial phase modulation |
| Binary2Phase | `d`, `order2`-`order12` | Polynomial phase to FFT intensity |
| Diffractive candidate | `phase_scale` | Simple phase-to-intensity fallback |

## Contract

- Output PSF shape is `[band_count, psf_size, psf_size]`.
- `psf_requires_grad=true` is required for success.
- PSF normalization stays inside torch autograd.
- `PSF_GRAPH_DISCONNECTED` is returned when no component parameter receives a gradient.
- This is not a full GeoLens PSF path.

## CLI

```bash
python -m optiresearch.cli run-component-surrogate-hsi-codesign \
  --component fresnel \
  --dataset synthetic \
  --steps 3 \
  --device cpu
```
