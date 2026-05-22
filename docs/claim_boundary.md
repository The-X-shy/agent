# Claim Boundary

Defines what the system CAN and CANNOT claim at the current evidence level.

## Categories

- **Supported Claims**: Claims backed by available evidence.
- **Qualified Claims**: Claims that require scope caveats (e.g., "mock only").
- **Unsupported Claims**: Claims that MUST NOT be made without further evidence.

## Rules

- Synthetic/mock results must not be written as real HSI performance.
- DeepLens adapter_proxy must not be written as native validation.
- Public dataset + mock optical must not be written as real camera experiment.
- Black-box search must not be written as differentiable optimization.
- Native optimization claims require gradient_norm > 0 AND parameters_changed = True.
- Native optimization for a single lens class does not imply all classes are differentiable.

## CLI

```bash
# Export full claim boundary
python -m optiresearch.cli export-claim-boundary

# Pre-check a specific claim (Phase 24)
python -m optiresearch.cli check-claim \
  --claim-text "Full DeepLens wave-optics native HSI co-design is supported" \
  --backend-id deeplens_geolens_geometric
```

Output: `workspace/reports/claim_boundary.md`, `workspace/reports/claim_boundary.json`

## Claim Gate v2 (Phase 24)

`ClaimGateV2` provides automated pre-check of claims before they enter the evidence system.
It detects 8 violation types: proxy_as_waveoptics, geometric_as_coherent, synthetic_as_real,
black_box_as_native, unsupported_path_as_supported, differentiable_as_improves,
rollback_protection_as_improvement, local_only_as_robust.

See [claim_gate_v2.md](claim_gate_v2.md) for details.
