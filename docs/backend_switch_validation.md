# Backend Switch Validation

## Concept

A backend switch is **triggered** when `claim_ceiling_reached` is detected.
A backend switch is **validated** when a lightweight probe on the new backend
succeeds. These are distinct states tracked separately.

## TrajectoryEvaluation Fields

| Field | Type | Meaning |
|---|---|---|
| `backend_switch_triggered` | bool | Any iteration has `switched_from_backend` in execution_result |
| `backend_switch_validated` | bool | Any iteration has `backend_switch_validated=True` in execution_result |
| `backend_probe_success` | bool | Any iteration's result_payload has `probe_status=succeeded` |
| `backend_probe_unavailable` | bool | Any iteration's result_payload has `probe_status=unavailable` |
| `evidence_gain_after_switch` | bool | Post-switch evidence levels contain levels not seen pre-switch |

## Validation Flow

```
1. claim_ceiling_reached on backend A
2. Loop switches to backend B
3. pending_backend_switch=True injected
4. StrategyEngine recommends probe_new_backend
5. Lightweight backend probe executes on backend B
6. If probe succeeds:
   - backend_switch_validated=True
   - pending_backend_switch cleared
   - Next iteration uses normal strategy flow on backend B
7. If probe fails:
   - Try alternative backend from progression graph
   - If no alternatives: stop with backend_switch_failed
```

## Alternative Backend Fallback

When the primary next backend's probe fails:

1. `get_all_edges_from(source_backend)` returns all possible target backends
2. Loop skips the failed backend and tries the next available one
3. Respects `max_backend_switches` limit from AutonomousLoopSpec
4. If all alternatives exhausted: `stop_reason=backend_switch_failed`

## Report Sections

The autonomous loop report includes:
- **Backend Progression**: Iter | Backend | Evidence Level | Validated
- **Backend Probe Results**: Iter | Backend | Probe Status | Probe Time (s)
- **Backend Switch Validation**: Switch Triggered | Switch Validated | Probe Success
