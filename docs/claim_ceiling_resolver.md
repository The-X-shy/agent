# Claim Ceiling Resolver

Phase 41 introduces the `ClaimCeilingResolver` to determine the final claim ceiling from multiple constraint sources.

## Problem

Before Phase 41, `ClaimGateV2._compute_max_allowed_claim()` used only the backend_id to look up `backend.claim_ceiling`. This meant a design with `backend_id=deeplens_geolens_geometric` would get `max_allowed_claim=native_lens_simulation` even when the actual handler only produced `lightweight_scientific_execution`.

## Solution

`resolve_claim_ceiling()` computes ceilings from four sources:

| Source | Meaning |
|---|---|
| Handler ceiling | From `HandlerCapabilityRegistry.max_claim_ceiling` |
| Backend ceiling | From backend registry `claim_ceiling` |
| Dataset ceiling | `lightweight_scientific_execution` for synthetic, `real_hsi_performance` for real |
| Execution fidelity ceiling | `lightweight_scientific_execution` for proxy, tiered for native |

Additional constraints: no physical backend, no native backend, proxy fallback, FFT proxy, evidence level.

## Final Ceiling

`final_claim_ceiling = min(handler, backend, dataset, fidelity)` by evidence rank.

The `ceiling_source` and `limiting_factor` fields identify which constraint was most restrictive.

## CLI

```bash
python -m optiresearch.cli resolve-claim-ceiling \
  --handler-id objective_redesign_simpler_metric \
  --backend-id deeplens_geolens_geometric \
  --dataset synthetic \
  --execution-fidelity lightweight_proxy
```
