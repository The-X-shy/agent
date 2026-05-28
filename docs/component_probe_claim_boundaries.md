# Component Probe Claim Boundaries

Component-level probes produce evidence scoped to individual DeepLens surface
components. This document defines what claims they can and cannot support.

## Allowed Claims

| Claim | Supported? |
|-------|-----------|
| Component is importable | Yes |
| Component is instantiatable | Yes |
| Component exposes trainable parameters | Yes |
| Component supports autograd (requires_grad + backward) | Yes |
| Component parameters change after optimizer step | Yes |
| `native_component_optimization` | Yes (ceiling) |

## Blocked Claims

| Claim | Why Blocked |
|-------|------------|
| `native_lens_optimization` | Component ≠ lens assembly |
| `full_geolens_direct_update` | Blocked route per Phase 61 |
| `hsi_improvement` | No HSI reconstruction in component probe |
| `real_camera_validation` | No physical measurement |
| `lens_level_psf_improvement` | Component-level only |

## Claim Ceiling

The **claim ceiling** for component probes is capped at `native_component_optimization`.
This is enforced by:

1. **Backend registry** (`deeplens_fresnel_component`, `deeplens_binary2phase_component` both
   have `claim_ceiling="native_component_optimization"`).
2. **Claim ceiling resolver** selects the minimum ceiling across all evidence sources.
3. **Claim gate** detects violations when component evidence is used for lens-level claims.
