# Handler Capability Registry

Phase 40 introduces the `HandlerCapabilityRegistry` as the single source of truth for what each execution handler can actually produce.

## Purpose

Before Phase 40, evidence levels were hardcoded across 4+ files:
- `ExperimentDesignGenerator` (sets expected evidence)
- `CandidatePlanEvaluator` (scores based on evidence)
- `ClaimGateV2` (checks claims against evidence)
- `SkillRegistryV2` (registers skills with evidence levels)

This caused mismatches — e.g., `objective_redesign_simpler_metric_mse_only` had `expected_evidence_level=native_lens_simulation` but the actual handler produced `lightweight_scientific_execution`.

The registry is the authoritative source that all other components query.

## Data Model

```python
@dataclass
class HandlerCapability:
    handler_id: str
    design_type: str              # scientific, probe, report, data_request
    task_type: str
    supported_execution_modes: list[str]  # dry_run, local, remote_opt_in
    actual_evidence_level: str
    max_claim_ceiling: str
    synthetic_only: bool
    native_backend_required: bool
    physical_backend: bool
    real_data_required: bool
    remote_required: bool
    metrics_supported: list[str]
    artifacts_supported: list[str]
    known_limitations: list[str]
    compatible_design_ids: list[str]
```

## Registered Handlers

| Handler | Type | Evidence | Locally Executable |
|---|---|---|---|
| `objective_redesign_simpler_metric` | scientific | `lightweight_scientific_execution` | Yes |
| `param_reduction_sweep` | scientific | `lightweight_scientific_execution` | Yes |
| `backend_switch_waveoptics_coherent` | probe | `structured_unsupported` | No |
| `report_negative_result_doc` | report | `report_only` | Yes |
| `real_data_request` | data_request | `requires_user_data` | No |

## CLI

```bash
python -m optiresearch.cli list-handler-capabilities
python -m optiresearch.cli inspect-handler-capability --handler-id objective_redesign_simpler_metric
```
