# Cross-Backend Trajectory Evaluation

## Overview

The trajectory evaluator tracks backend history and evidence level progression
across iterations, enabling multi-backend loops.

## New Fields

| Field | Type | Description |
|-------|------|-------------|
| `backend_history` | `list[str]` | Backend IDs in order of iteration |
| `backend_switch_count` | `int` | Number of backend switches detected |
| `evidence_level_progression` | `bool` | True when multiple evidence levels seen |

## Phase 31: Switch Validation Fields

| Field | Type | Description |
|-------|------|-------------|
| `backend_switch_triggered` | `bool` | Any iteration has `switched_from_backend` in execution_result |
| `backend_switch_validated` | `bool` | Any iteration has `backend_switch_validated=True` |
| `backend_probe_success` | `bool` | Probe on new backend returned `probe_status=succeeded` |
| `backend_probe_unavailable` | `bool` | Probe returned `probe_status=unavailable` |
| `evidence_gain_after_switch` | `bool` | Post-switch evidence levels contain new levels not seen pre-switch |

## Claim Ceiling Detection

`claim_ceiling_reached` now requires:
- 2+ iterations
- All iterations on the **same** backend
- All iterations with the **same** `max_allowed_claim`

When different backends produce different claim ceilings, this is detected as
`evidence_level_progression` instead.

## Caveats

- Metrics from different backends may not be directly comparable.
- Cross-backend comparison should prioritize evidence level / claim gain over
  raw metric values.
- A backend switch that increases evidence level is considered progress even
  if the metric value is numerically worse.
- A backend switch that fails is recorded as a negative result.
