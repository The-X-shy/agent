# Surrogate PSF vs GeoLens PSF

The component surrogate PSF and the full GeoLens PSF answer different
questions.

| Path | Purpose | Gradient status | Claim scope |
|---|---|---|---|
| Component surrogate PSF | Test component-to-HSI gradient flow | Differentiable | `component_surrogate_hsi_codesign` |
| Full GeoLens geometric PSF | Native lens assembly simulation | Blocked in Phase 60-61 | Diagnostic boundary only |

## Important Distinction

The surrogate PSF uses component parameter semantics from Fresnel and
Binary2Phase, then constructs a differentiable kernel in torch. It does not
call the full GeoLens direct update route and does not validate lens-level
physical performance.

The full GeoLens route remains blocked until its PSF path exposes trainable
parameters and a connected autograd graph.
