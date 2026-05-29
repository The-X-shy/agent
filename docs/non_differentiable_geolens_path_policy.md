# GeoLens Geometric PSF Training Policy

Phase 60-61 originally reported that DeepLens GeoLens on WSL had:
- `parameter_count=0`
- `trainable_param_count=0`
- `graph_connected=false`
- `psf_requires_grad=false`

That conclusion was caused by the OptiResearch wrapper using the wrong API:
`GeoLens` is not an `nn.Module`, so `geolens.parameters()` is not the source of
trainable lens parameters. The native DeepLens route is
`get_optimizer_params()` / `get_optimizer()`.

The geometric PSF path also needs float32 tensors. Forcing float64 can break
DeepLens geometric PSF execution on WSL.

## Policy

### Conditional Native Route
- **`full_geolens_direct_update`** is allowed only when the native audit proves:
  - `trainable_param_count > 0`
  - `params_with_grad > 0`
  - `psf_requires_grad=true`
  - `loss_requires_grad=true`
  - `graph_connected=true`
  - a candidate update changes at least one GeoLens parameter

When these checks pass, the claim ceiling is still only
`native_lens_simulation`.

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

DeepLens exposes GeoLens optimization through native optimizer parameter groups,
not through the standard PyTorch module parameter API. OptiResearch must
activate those native parameter groups before judging whether the geometric PSF
path is differentiable.

Component surrogate HSI co-design is allowed because it does not execute
`full_geolens_direct_update` and does not claim full GeoLens success. It remains
a separate, lower-ceiling route.

The coherent/full wave-optics path remains unverified for this fix.

## Evidence

- Phase 60: old diagnostics reported `parameter_count=0` because they used
  `geolens.parameters()`
- Phase 64 fix: diagnostics now use `get_optimizer_params()` / `get_optimizer()`
  and float32 geometric PSF
- WSL diagnostic evidence: full GeoLens geometric path exposes trainable
  parameters and connected gradients when audited through the native API
- Phase 62: Component backends validated independently
- Phase 63: Component surrogate PSF HSI loop validates gradient flow to
  component parameters under synthetic HSI loss
