# Trajectory Evaluator (Phase 25)

`evaluate_trajectory()` analyzes iteration trajectories to determine whether
the autonomous loop should continue or stop.

## Stop Conditions

| Condition | Trigger |
|-----------|---------|
| `no_iterations` | Zero iterations completed |
| `claim_ceiling_reached` | All iterations hit same claim ceiling for >=2 iterations |
| `repeated_failure` | >=2 consecutive failed/skipped/unsupported iterations |
| `max_iterations_reached` | Iteration count reached spec.max_iterations |
| `no_improvement` | Primary metric did not improve over the trajectory |

## Detection Logic

1. **Improvement**: Best metric value is better than the first iteration's
2. **Repeated failure**: >=2 consecutive iterations with status `failed`, `skipped`, or `unsupported`
3. **Claim ceiling**: All iterations have the same `max_allowed_claim` value

## Primary Metric Extraction

Prefers in order: `reconstruction_loss_after`, `loss_after`, `mse_after`.
Lower values are considered better.

## Programmatic API

```python
from optiresearch.agents.trajectory_evaluator import evaluate_trajectory

eval_result = evaluate_trajectory(iterations, spec)
print(eval_result.improvement_detected)
print(eval_result.best_iteration)
print(eval_result.stop_reason)
```
