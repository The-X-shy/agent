# Claim Downgrade Semantics

## Overview

In the autonomous research loop, a **claim downgrade** is a safety wording
correction — NOT an experimental failure. This distinction is critical for
multi-iteration loops that must continue producing metrics after a downgrade.

## Execution Status vs. Claim Status

| Concept | Meaning | Loop Behavior |
|---------|---------|---------------|
| `execution status = succeeded` | Experiment ran successfully, produced metrics | Continue |
| `execution status = claim_downgraded` | Experiment ran but claim was capped | Continue with metrics |
| `claim status = qualified` | Claim gate downgraded wording | Record, continue |
| `claim status = unsupported` | Claim violates ceiling (fatal) | Use safe wording, continue |
| `execution status = failed` | Experiment crashed or errored | Count as failure |

## Key Principles

1. **claim_downgraded is not a failure.** It does not count toward
   `repeated_failure_detected` in the trajectory evaluator.

2. **Metrics from downgraded experiments are valid.** If the experiment
   produced a `result_payload` with `reconstruction_loss_after`, those
   metrics enter the metric trajectory.

3. **Claim downgrade does not trigger early stop.** The loop continues
   as long as metrics are produced and `min_iterations_before_stop` is
   not yet satisfied.

4. **Downgraded claims still produce evidence.** The evidence level is
   capped to the backend's claim ceiling, but the experimental results
   are still recorded in research memory.

## How It Works

### Backend Evidence Caps

Each backend has an allowlist of task types with associated evidence caps:

| Backend | Task | Evidence Cap |
|---------|------|-------------|
| `phase_to_fft_proxy` | `stable_lens_hsi_codesign` | `native_full_reconstruction_proxy` |
| `phase_to_fft_proxy` | `native_hsi_codesign` | `native_hsi_proxy` |
| `deeplens_geolens_geometric` | `stable_lens_hsi_codesign` | `native_lens_simulation` |

### Experiment Controller

The controller checks `get_backend_task_evidence_cap()` to determine if a
task is allowed on a backend. If allowed, the evidence level is capped
and the experiment runs. If not allowed, the controller returns `unsupported`.

### Claim Gate

The ClaimGateV2 still performs its full violation detection. It may return
`qualified` or `unsupported` decisions. In the loop, both trigger
`claim_downgraded = True` on the execution result, but neither stops the
loop by itself.

### Trajectory Evaluator

The trajectory evaluator tracks `claim_downgraded_count` separately from
failure counts. Claim downgrades do not increment the `consecutive_failures`
counter and do not trigger `repeated_failure_detected`.

## Stop Conditions That Still Apply

The loop will still stop if:
- `max_iterations` is reached
- `repeated_failure_detected` (2+ consecutive actual failures)
- `claim_ceiling_reached` (all iterations at same ceiling with 2+ iterations)
- `no_improvement` after `min_iterations_before_stop` + `no_improvement_patience`
