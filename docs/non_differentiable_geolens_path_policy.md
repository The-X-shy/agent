# Non-Differentiable GeoLens Path Policy

Phase 60-61 confirmed that DeepLens GeoLens on WSL has:
- `parameter_count=0`
- `trainable_param_count=0`
- `graph_connected=false`
- `psf_requires_grad=false`

This document defines the policy for handling this situation.

## Policy

### Blocked
- **`full_geolens_direct_update`** — permanently blocked as a primary strategy.
  The GeoLens geometric PSF path does not expose trainable parameters and cannot
  support gradient-based optimization.

### Allowed
- **Component-level probes** — Fresnel, Binary2Phase, and other surface
  components are independently validated for differentiability.
- **Component surrogate HSI co-design** — validated component parameter
  semantics may be wired into a differentiable surrogate PSF and synthetic HSI
  reconstruction loop.

### Conditional
- **GeoLens curriculum probe** — diagnostic only; cannot produce optimization
  improvements.
- **GeoLens regularized probe** — diagnostic only; tests gradient stabilization
  strategies without claiming improvement.

## Rationale

The GeoLens API constructs its geometric PSF via non-differentiable ray tracing.
While the individual surface components (Fresnel, Binary2Phase) may be
differentiable, the assembled GeoLens does not propagate gradients through
the full pipeline. Until DeepLens provides a differentiable PSF path through
GeoLens, lens-level optimization must rely on component-level strategies or
await upstream API changes.

Component surrogate HSI co-design is allowed because it does not execute
`full_geolens_direct_update` and does not claim full GeoLens success.

## Evidence

- Phase 60: GeoLens `parameter_count=0` confirmed via `inspect_deeplens_trainable_parameters()`
- Phase 60: GeoLens `graph_connected=false` confirmed via `run_deeplens_autograd_audit()`
- Phase 60: GeoLens `psf_requires_grad=false` confirmed via PSF tensor inspection
- Phase 62: Component backends validated independently
- Phase 63: Component surrogate PSF HSI loop validates gradient flow to
  component parameters under synthetic HSI loss
