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
