# Metric-First Trajectory Evaluation

## Overview

The trajectory evaluator (`evaluate_trajectory()`) has been updated to use
a **metric-first** approach for stop decisions. Instead of stopping on the
first sign of non-improvement, it requires a minimum number of iterations
and a patience window before emitting a `no_improvement` stop.

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_iterations_before_stop` | 2 | Minimum iterations before any stop reason can be emitted (except repeated_failure) |
| `no_improvement_patience` | 2 | Consecutive non-improving iterations required before no_improvement stop |

## Algorithm

1. Extract primary metric from each iteration's `result_payload` (prefers
   `reconstruction_loss_after` > `loss_after` > `mse_after`).

2. Build `metric_trajectory` list with all extracted values.

3. Compute `improvement_detected`: `best_val < trajectory[0]` when >= 2 iterations.

4. Count `no_improvement_streak`: consecutive iterations where `val >= best_so_far`
   (no improvement over the best metric seen so far).

5. Set stop reason using priority:
   - `claim_ceiling_reached`
   - `repeated_failure` (only when also `not improvement`)
   - `max_iterations_reached`
   - `no_improvement` (only when `min_iterations_satisfied` AND `no_improvement_streak >= patience`)

## Example

```
Iteration 1: loss=0.05  → best_so_far=0.05, streak=0
Iteration 2: loss=0.06  → best_so_far=0.05, streak=1
Iteration 3: loss=0.04  → best_so_far=0.04, streak=0 (improvement!)
Iteration 4: loss=0.05  → best_so_far=0.04, streak=1
Iteration 5: loss=0.06  → best_so_far=0.04, streak=2 → stop_reason="no_improvement"
```

With `min_iterations_before_stop=2` and `no_improvement_patience=2`:
- After iteration 1: no stop (min_iterations not satisfied)
- After iteration 2: no stop (streak=1 < patience=2)
- After iteration 3: no stop (improvement detected, streak reset)
- After iteration 4: no stop (streak=1 < patience=2)
- After iteration 5: stop with "no_improvement"

## CLI Usage

```bash
python -m optiresearch.cli run-autonomous-research-loop-v2 \
  --objective "test multi-iteration" \
  --execution-mode local \
  --max-iterations 5 \
  --min-iterations-before-stop 2 \
  --no-improvement-patience 2
```
