# Evidence Level Alignment

Phase 40 ensures that designs' `expected_evidence_level` matches what the handler can actually produce.

## The Problem

| Design | Strategy Evidence | Handler Evidence | Status |
|---|---|---|---|
| `objective_redesign_simpler_metric_mse_only` | `native_lens_simulation` | `lightweight_scientific_execution` | **Downgraded** |
| `backend_switch_waveoptics_coherent` | `native_waveoptics_simulation` | `structured_unsupported` | **Downgraded** |
| `param_reduction_sweep` | `native_lens_simulation` | `lightweight_scientific_execution` | **Downgraded** |
| `report_negative_result_doc` | `negative_result` | `report_only` | Aligned |

## How It Works

1. `ExperimentDesignGenerator` generates designs with strategy-level evidence levels
2. `_align_evidence_levels()` queries `HandlerCapabilityRegistry` for each design
3. If handler actual evidence < strategy expected evidence → `downgraded_to_handler_capability`
4. If handler actual evidence == strategy expected → `aligned`
5. If no handler found → `unsupported`

## Alignment Statuses

| Status | Meaning |
|---|---|
| `aligned` | Handler produces exactly the expected evidence |
| `downgraded_to_handler_capability` | Strategy wanted higher evidence, handler limited it |
| `unsupported` | No handler exists for this design |

## Scoring Impact

`CandidatePlanEvaluator` uses `actual_handler_evidence_level` (not `expected_evidence_level`) for evidence gain scoring. Downgraded designs receive a -0.05 penalty but remain executable.

## ClaimGate Impact

New violations:
- `evidence_level_overestimated` — when expected > actual evidence
- `handler_capability_exceeded` — when claim exceeds handler ceiling
