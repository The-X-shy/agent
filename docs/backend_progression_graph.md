# Backend Progression Graph

## Overview

The `BackendProgressionGraph` defines valid backend-to-backend transitions for
multi-backend autonomous loops. When the current backend's claim ceiling is
reached, the loop queries this graph to find the next higher-evidence backend.

## Default Progression Edges

| From | To | Reason | Claim Gain | Cost |
|------|----|--------|------------|------|
| `mock_deeplens` | `phase_to_fft_proxy` | Mock to differentiable proxy | mock_simulation → native_full_reconstruction_proxy | low |
| `phase_to_fft_proxy` | `deeplens_geolens_geometric` | FFT proxy to native lens simulation | native_full_reconstruction_proxy → native_lens_simulation | requires_deeplens |
| `phase_to_fft_proxy` | `deeplens_fresnel_component` | FFT proxy to native component | native_full_reconstruction_proxy → native_component_optimization | requires_deeplens |
| `deeplens_fresnel_component` | `deeplens_geolens_geometric` | Component to lens-file simulation | native_component_optimization → native_lens_simulation | requires_deeplens |
| `deeplens_geolens_geometric` | `deeplens_coherent_asm` | Lens simulation to wave-optics probe | native_lens_simulation → waveoptics_probe | requires_deeplens |
| `local_synthetic_hsi` | `phase_to_fft_proxy` | Synthetic to differentiable proxy | synthetic_hsi_simulation → native_full_reconstruction_proxy | low |

## API

```python
from optiresearch.backends.progression import get_next_backend, list_progression_from

# Get the recommended next backend
result = get_next_backend("phase_to_fft_proxy", reason="claim_ceiling_reached")
# => {"next_backend": "deeplens_geolens_geometric", "expected_claim_gain": "...", ...}

# List all possible next backends
next_backends = list_progression_from("phase_to_fft_proxy")
# => ["deeplens_geolens_geometric", "deeplens_fresnel_component"]
```
