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

## CLI

```bash
python -m optiresearch.cli export-claim-boundary
```

Output: `workspace/reports/claim_boundary.md`, `workspace/reports/claim_boundary.json`
