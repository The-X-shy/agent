# Component-Level Pivot Strategy

After Phase 60-61, the system added a component-level fallback for cases where
full GeoLens autograd audit fails. The original `parameter_count=0` finding was
later corrected: full GeoLens geometric training must be audited through
DeepLens `get_optimizer_params()` / `get_optimizer()`, not through
`geolens.parameters()`.

## Strategy

When full GeoLens audit is unavailable or disconnected, the system probes
individual DeepLens surface components:

1. **Fresnel** — diffractive surface with trainable `f0` parameter
2. **Binary2Phase** — phase surface with 7 trainable order coefficients

## Recovery Chain

```
GeoLens autograd audit failure
  → Gradient instability diagnosis
  → Recovery policy ranks component_first_fresnel_probe (score 9)
  → Evidence strategy reasoner generates component_first strategy
  → Experiment design generator creates component probe designs
  → Agent plan execution loop dispatches to run_deeplens_component_probe()
  → If component probes succeed, run component_surrogate_hsi_codesign
```

## Blocked Routes

- `full_geolens_direct_update` — blocked only when the native optimizer audit
  fails or is unavailable.
- If the audit passes with connected gradients and a parameter update, the route
  may run at `native_lens_simulation` claim ceiling.

## Phase 63 Extension

Phase 63 adds a minimal component-level HSI co-design loop. It uses validated
Fresnel/Binary2Phase parameter semantics to build a differentiable surrogate
PSF, runs synthetic HSI reconstruction, and backpropagates reconstruction loss
to component parameters.

Claim ceiling: `component_surrogate_hsi_codesign`.

This is still not full GeoLens lens-level optimization. It is the fallback when
the native full GeoLens audit does not pass.

## Related

- `docs/deeplens_component_first_strategy.md`
- `docs/non_differentiable_geolens_path_policy.md`
- `docs/component_surrogate_hsi_codesign.md`
