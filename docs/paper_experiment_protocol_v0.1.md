# Paper Experiment Protocol v0.1

This protocol freezes the evidence boundaries for OptiResearch Agent paper experiments.

## 1. System-level experiments

- memory ablation
- skill ablation
- claim evidence rate
- unsupported claim rate

## 2. Optical backend experiments

- mock baseline
- DeepLens smoke
- DeepLens adapter_proxy
- DeepLens semi_native
- wavelength-aware PSF contract

## 3. HSI experiments

- synthetic optical-sensitive HSI
- reconstructor matrix
- public/local dataset matrix
- DeepLens PSF compatibility

## 4. Evidence level definitions

| Evidence level | Meaning |
|---|---|
| `mock` | synthetic or mock-only result |
| `deeplens_smoke` | DeepLens import/API smoke validation only |
| `deeplens_adapter_proxy` | DeepLens base output plus adapter-level proxy behavior |
| `deeplens_semi_native` | partial native DeepLens behavior, not full native optimization |
| `synthetic_hsi` | synthetic HSI dataset evaluation |
| `public_hsi_mock` | public/local HSI data with mock optical measurement |
| `public_hsi_deeplens_proxy` | public/local HSI data with DeepLens adapter-proxy optical backend |
| `public_hsi_deeplens_semi_native` | public/local HSI data with semi-native DeepLens backend |
| `native_optimized` | native optimized optical design, not yet established by default |
| `real_lab` | measured real camera/lab experiment |

## 5. Claim rules

Allowed:

- synthetic HSI claims only inside synthetic scope;
- public/local HSI reconstruction claims only when dataset manifest, matrix result, backend, and reconstructor are declared;
- DeepLens adapter_proxy claims only as adapter-proxy claims;
- semi_native claims only as semi-native claims.

Not allowed:

- synthetic/mock results as real HSI performance;
- public/local dataset plus mock optical encoder as real camera validation;
- DeepLens adapter_proxy as native physical validation;
- LLM-generated status as final evidence status.

ClaimEvidence is the final gate for all claim status.

## Phase 13 Status

As of Phase 13, the protocol is frozen and the complete paper evidence package is available. See:
- `workspace/final_paper_package/` for the full reproducibility package
- `docs/claim_boundary.md` for claim boundaries
- `docs/final_benchmark.md` for the benchmark registry

