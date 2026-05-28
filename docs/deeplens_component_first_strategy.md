# DeepLens Component-First Strategy

The `component_first` strategy family probes individual DeepLens surface
component classes before attempting full lens-level optimization.

## Strategy Family

| Strategy | Component | Surface Class | Skill |
|----------|-----------|--------------|-------|
| `component_first_fresnel_probe` | Fresnel | `Fresnel` | `deeplens_component_first_probe` |
| `component_first_binary2phase_probe` | Binary2Phase | `Binary2Phase` | `deeplens_component_first_probe` |
| `diffractive_component_probe` | Diffractive | `Fresnel` (candidate) | `deeplens_component_first_probe` |

## Design Scoring

Component-first designs are scored by `CandidatePlanEvaluator`:
- **Feasibility:** 9 (high — minimal dependencies)
- **Evidence gain:** moderate (diagnostic_evidence ceiling)
- **Risk:** low (isolated component, no HSI pipeline)

## Execution

Component-first designs are classified as diagnostic by `_is_diagnostic_design()`
and dispatched to `run_deeplens_component_probe()` both locally and remotely.

## Claim Boundary

- **Max claim:** `native_component_optimization`
- **Min claim:** `diagnostic_evidence`
- **Not supported:** lens optimization, HSI performance, real camera

See `docs/component_probe_claim_boundaries.md` for the full claim table.
