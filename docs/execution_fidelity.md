# Execution Fidelity

## Purpose

Distinguishes between lightweight proxy experiments and full native DeepLens
experiments, preventing proxy results from being claimed as native GeoLens
geometric PSF results.

## Fidelity Levels

| Level | Description | PSF Source | Backend |
|---|---|---|---|
| `lightweight_proxy` | FFT-based Fraunhofer proxy | `torch.fft.fft2(exp(i*phase))` | Any |
| `deeplens_native_geometric` | Actual GeoLens geometric ray-tracing | `geolens.psf(model="geometric")` | `deeplens_geolens_geometric` |
| `deeplens_native_waveoptics` | Coherent wave-optics propagation | `geolens.psf(model="coherent")` | `deeplens_coherent_asm` (not differentiable) |

## ExperimentSpecV2 Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `execution_fidelity` | str | `""` | Which fidelity level to enforce |
| `allow_proxy_fallback` | bool | `True` | Allow fallback to proxy if native unavailable |
| `require_deeplens_native` | bool | `False` | Fail if native DeepLens not available |

## ClaimGate Integration

The `proxy_as_native_geolens` violation (Phase 33) prevents claims mentioning
"native lens" or "geolens" from passing when `execution_fidelity=lightweight_proxy`
or `phase_to_fft_proxy_used=True` on `deeplens_geolens_geometric` backend.

## Controller Routing

`native_lens_simulation_codesign` routes to:
- `_run_stable_lens_hsi` (full native) when `backend_id == "deeplens_geolens_geometric"`
- `_run_lightweight_stable_lens_hsi` (proxy) for all other backends
