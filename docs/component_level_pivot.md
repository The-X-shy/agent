# Component-Level Pivot Strategy

After Phase 60-61 confirmed that GeoLens autograd fails (parameter_count=0,
graph_connected=false), the system pivots from lens-level optimization to
component-level validation.

## Strategy

Instead of optimizing `GeoLens.psf()` directly (which has no trainable
parameters), the system probes individual DeepLens surface components:

1. **Fresnel** — diffractive surface with trainable `f0` parameter
2. **Binary2Phase** — phase surface with 7 trainable order coefficients

## Recovery Chain

```
GeoLens autograd failure
  → Gradient instability diagnosis
  → Recovery policy ranks component_first_fresnel_probe (score 9)
  → Evidence strategy reasoner generates component_first strategy
  → Experiment design generator creates component probe designs
  → Agent plan execution loop dispatches to run_deeplens_component_probe()
  → If component probes succeed, run component_surrogate_hsi_codesign
```

## Blocked Routes

- `full_geolens_direct_update` — permanently blocked until GeoLens exposes
  trainable parameters and a connected autograd graph.

## Phase 63 Extension

Phase 63 adds a minimal component-level HSI co-design loop. It uses validated
Fresnel/Binary2Phase parameter semantics to build a differentiable surrogate
PSF, runs synthetic HSI reconstruction, and backpropagates reconstruction loss
to component parameters.

Claim ceiling: `component_surrogate_hsi_codesign`.

This is still not full GeoLens lens-level optimization.

## Related

- `docs/deeplens_component_first_strategy.md`
- `docs/non_differentiable_geolens_path_policy.md`
- `docs/component_surrogate_hsi_codesign.md`
