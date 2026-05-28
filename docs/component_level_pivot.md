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
```

## Blocked Routes

- `full_geolens_direct_update` — permanently blocked until GeoLens exposes
  trainable parameters and a connected autograd graph.

## Related

- `docs/deeplens_component_first_strategy.md`
- `docs/non_differentiable_geolens_path_policy.md`
