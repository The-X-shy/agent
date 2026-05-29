# Surrogate PSF vs GeoLens PSF

The component surrogate PSF and the full GeoLens PSF answer different
questions.

| Path | Purpose | Gradient status | Claim scope |
|---|---|---|---|
| Component surrogate PSF | Test component-to-HSI gradient flow | Differentiable | `component_surrogate_hsi_codesign` |
| Full GeoLens geometric PSF | Native lens assembly simulation | Conditional: requires native optimizer audit | `native_lens_simulation` |

## Important Distinction

The surrogate PSF uses component parameter semantics from Fresnel and
Binary2Phase, then constructs a differentiable kernel in torch. It does not
call the full GeoLens direct update route and does not validate lens-level
physical performance.

The full GeoLens geometric route is separate. It may be trained only after
`get_optimizer_params()` / `get_optimizer()` expose trainable parameters and the
float32 geometric PSF audit proves a connected autograd graph.
