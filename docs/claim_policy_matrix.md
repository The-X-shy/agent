# Claim Policy Matrix

The Claim Policy Matrix maps each evidence level to its supported claims, blocked claims, safe wording, and artifact/metric requirements.

## Evidence Level Hierarchy

| Level | Rank | Supported Claims |
|---|---|---|
| unsupported | 0 | None |
| report_only | 1 | Documentation and traceability |
| mock_simulation | 2 | Mock pipeline integration correctness |
| deeplens_integration_smoke | 3 | DeepLens API integration verified |
| native_component_optimization | 4 | Component-level gradient optimization |
| component_surrogate_hsi_codesign | 5 | Component surrogate HSI co-design synthetic |
| native_hsi_proxy | 5 | HSI proxy optimization |
| lightweight_scientific_execution | 7 | Synthetic lightweight HSI co-design |
| native_lens_simulation | 8 | Native lens simulation synthetic HSI |
| native_waveoptics_simulation | 9 | Native wave-optics simulation |
| stable_native_lens_hsi_codesign | 10 | Stable reproducible native lens HSI |
| rollback_protected_native_lens_hsi | 11 | Rollback-protected stability-verified HSI |
| real_hsi_performance | 12 | Real HSI validated on physical measurements |

## Key Rules

1. Synthetic results MUST NOT be written as real HSI performance
2. Component-level results MUST NOT be written as full lens validation
3. Geometric optics MUST NOT be written as wave-optics
4. Every evidence level has a safe_wording_template
5. Higher evidence levels support all claims from lower levels

## Output

```bash
python -m optiresearch.cli export-claim-policy-matrix
```

Outputs to `workspace/system_capability/claim_policy_matrix.{json,csv,md}`
