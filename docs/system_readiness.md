# System Readiness Levels

## Capability Maturity Levels

| Level | Criteria |
|---|---|
| experimental | Capability exists but not validated |
| validated_local | Works in local execution |
| validated_remote | Works in remote (WSL) execution |
| benchmarked | Passed multi-config reproducibility benchmark |
| production_ready | Ready for continuous operation |

## Current System Readiness

The overall system readiness score is computed from:
- Contract coverage across all dimensions
- Test coverage proxy per handler
- Documentation coverage proxy per handler

Current score and detailed breakdown are available via:
```bash
python -m optiresearch.cli export-contract-coverage-dashboard
```

## What's Ready

- Handler capability registry with 13+ handlers
- Optical backend registry with 9 backends
- ClaimGate V2 with 16 evidence levels
- Remote WSL execution infrastructure
- Native GeoLens geometric optimization
- Component backend Fresnel/Binary2Phase support

## What's Needed

- Real HSI validation with physical camera data
- Wave-optics coherent propagation validation
- Cross-backend benchmark matrix
- Production deployment readiness assessment
