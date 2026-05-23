# Post-Switch Strategy Context

## Problem

After a backend switch is triggered by `claim_ceiling_reached`, the next
iteration enters the strategy phase with no awareness of the switch. The
StrategyEngine sees a stale `claim_ceiling_reached` signal or falls through
to `stop_and_report`, producing `strategy_could_not_map_to_experiment`.

## Solution

Three layers of context injection:

1. **Switch-time injection**: When `claim_ceiling_reached` triggers a backend
   switch, the loop writes `pending_backend_switch=True` along with
   `switched_from_backend` and `switched_to_backend` into the iteration's
   `execution_result`. This ensures the next iteration's strategy phase
   receives the switch context.

2. **Strategy recognition**: A new highest-priority rule in StrategyEngine
   (`pending_backend_switch_probe`) fires when `pending_backend_switch=True`,
   recommending `probe_new_backend`. This overrides the stale
   `claim_ceiling_reached` signal.

3. **LLM awareness**: The LLM planner prompt includes a new Rule 17 and a
   dynamic "Pending Backend Switch Detected" section when switch context is
   present in recent results.

## Flow

```
Iteration N:   claim_ceiling_reached detected
               -> backend_id switches from A to B
               -> pending_backend_switch=True injected into execution_result

Iteration N+1: previous = iterations[N].execution_result
               -> contains pending_backend_switch=True
               -> StrategyEngine rules: probe_new_backend
               -> compile_experiment_spec: backend_probe task
               -> execute: lightweight backend probe
               -> if probe succeeds: backend_switch_validated=True
               -> if probe fails: try alternative backend or stop

Iteration N+2: normal experiment flow on validated backend
```

## Key Design Decisions

- Context flows through `execution_result` dict (no new LoopState class)
- `probe_new_backend` is the highest-priority rule in StrategyEngine
- After probe success, `pending_backend_switch` is cleared, preventing re-trigger
- Feedback context builder preserves switch fields for LLM consumption
