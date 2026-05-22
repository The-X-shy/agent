# Backend Registry

The optical backend registry provides a unified catalog of all available
optical simulation and optimization backends.

## Registered Backends

| backend_id | type | diff_level | claim_ceiling |
|---|---|---|---|
| mock_deeplens | mock | none | mock_simulation |
| phase_to_fft_proxy | proxy | differentiable_proxy | native_full_reconstruction_proxy |
| deeplens_fresnel_component | deeplens | native_component | native_component_optimization |
| deeplens_binary2phase_component | deeplens | native_component | native_component_optimization |
| deeplens_geolens_geometric | deeplens | native_lens_simulation | native_lens_simulation |
| deeplens_coherent_asm | deeplens | none | native_lens_simulation |
| deeplens_blackbox_source_psf | deeplens | black_box | deeplens_integration_smoke |
| local_synthetic_hsi | synthetic | differentiable_proxy | synthetic_hsi_simulation |

## Claim Ceiling Hierarchy

1. `unsupported`
2. `mock_simulation`
3. `deeplens_integration_smoke`
4. `native_component_optimization`
5. `native_hsi_proxy`
6. `native_full_reconstruction_proxy`
7. `native_lens_simulation`
8. `native_waveoptics`
9. `stable_native_lens_hsi_codesign`
10. `rollback_protected_native_lens_hsi`
11. `real_hsi_performance`

## Key Constraints

- `phase_to_fft_proxy` cannot exceed `native_full_reconstruction_proxy`
- `deeplens_geolens_geometric` cannot claim wave-optics
- `deeplens_coherent_asm` is NOT differentiable (requires_grad=False)
- `local_synthetic_hsi` cannot support real HSI performance claims

## CLI

```bash
python -m optiresearch.cli list-optical-backends
python -m optiresearch.cli inspect-optical-backend --backend-id deeplens_geolens_geometric
```

## Programmatic API

```python
from optiresearch.backends.registry import list_backends, get_backend

for b in list_backends():
    print(f"{b.backend_id}: {b.claim_ceiling}")

backend = get_backend("deeplens_geolens_geometric")
print(backend.known_failure_modes)
```
