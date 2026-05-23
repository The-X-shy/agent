# Validated Backend Experiment

## Purpose

After a backend probe succeeds and `post_probe_continuation_required` is set,
the next iteration runs a lightweight experiment on the validated backend to
produce actual task metrics (reconstruction loss, gradient norms).

## Task Type: `native_lens_simulation_codesign`

A new task type added in Phase 32. Routes to the lightweight stable lens HSI
co-design experiment via `ExperimentControllerV2._run_lightweight_stable_lens_hsi()`.

### Parameters

| Param | Default | Description |
|---|---|---|
| `max_steps` | 3 | Minimal steps for quick validation |
| `lightweight_mode` | True | Uses FFT proxy, no DeepLens required |
| `device` | cpu | Local execution |
| `rollback_on_loss_increase` | True | Safety mechanism |

### Evidence Level

`native_lens_simulation` when running against `deeplens_geolens_geometric`

### Registry Entry

```python
"deeplens_geolens_geometric": {
    "native_lens_simulation_codesign": "native_lens_simulation",
}
```

## Strategy Integration

1. `StrategyEngine` rule `post_probe_continuation` fires when
   `post_probe_continuation_required=True`
2. Action: `run_validated_backend_experiment`
3. `strategy_to_spec._pick_continuation_task(backend_id)` selects task type
4. `_build_payload` sets `max_steps=3`, `lightweight_mode=True`, `rollback_on_loss_increase=True`

## LLM Prompt (Rule 18)

The LLM planner is instructed to propose `run_validated_backend_experiment`
when `post_probe_continuation_required=True` in recent results, with the
validated backend's evidence ceiling as the claim boundary.
