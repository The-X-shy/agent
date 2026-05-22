# Claim Gate v2

`ClaimGateV2` pre-checks proposed scientific claims before they enter the
evidence system. It detects 8 violation types and enforces backend claim
ceilings.

## Violation Types

| Violation | Description | Decision |
|-----------|-------------|----------|
| proxy_as_waveoptics | Proxy FFT claiming wave-optics | unsupported |
| geometric_as_coherent | Geometric PSF claiming coherent | unsupported |
| synthetic_as_real | Synthetic data claiming real | unsupported |
| black_box_as_native | Black-box claiming native grad | unsupported |
| unsupported_path_as_supported | Broken path claiming supported | unsupported |
| differentiable_as_improves | Differentiability claiming better | qualified |
| rollback_protection_as_improvement | Rollback claiming improvement | qualified |
| local_only_as_robust | Local-only claiming robust | needs_followup |

## Decision Levels

- **supported**: No violations detected
- **qualified**: Claim needs caveats or rewording
- **needs_followup**: Additional evidence required
- **unsupported**: Claim exceeds evidence ceiling

## Safe Wording

When a violation is detected, `ClaimGateV2` generates safe alternative wording:

```
Input:  "Full DeepLens wave-optics native HSI co-design is supported"
Output: "Full DeepLens geometric ray-tracing native HSI co-design is
         partially supported (see caveats) [evidence ceiling:
         native_lens_simulation]"
```

## CLI

```bash
python -m optiresearch.cli check-claim \
  --claim-text "Full DeepLens wave-optics native HSI co-design is supported" \
  --backend-id deeplens_geolens_geometric
```

## Programmatic API

```python
from optiresearch.memory.claim_gate_v2 import ClaimGateV2

gate = ClaimGateV2()
decision = gate.check_claim(
    "My scientific claim",
    "deeplens_geolens_geometric",
    experiment_result={"reconstruction_loss_after": 0.5},
)

if decision.decision == "unsupported":
    print(f"Violation: {decision.violation_type}")
    print(f"Safe wording: {decision.safe_wording}")
```
