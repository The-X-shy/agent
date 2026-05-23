# Post-Probe Continuation

## Problem

After a backend switch probe succeeds, the next iteration has no clear path
forward. `backend_switch_validated` is set to `True` but no StrategyEngine
rule reads it. The engine falls through to `stop_and_report`.

## Solution

A new `post_probe_continuation_required` signal is injected into the
iteration's `execution_result` when a probe succeeds. This signal triggers
the `run_validated_backend_experiment` action, which maps to a lightweight
experiment on the validated backend.

## Flow

```
Iteration N:   probe_new_backend executes, probe succeeds
               -> backend_switch_validated = True
               -> pending_backend_switch = False
               -> post_probe_continuation_required = True
               -> validated_backend_id = <new backend>
               -> validated_backend_evidence_level = <evidence cap>

Iteration N+1: previous carries post_probe_continuation_required=True
               -> StrategyEngine: run_validated_backend_experiment
               -> compile: task type based on validated backend
               -> execute: lightweight experiment produces metrics
               -> if succeeded: post_probe_continuation_required = False

Iteration N+2: normal strategy flow on validated backend
```

## Related Actions

| Action | When | Maps To |
|--------|------|---------|
| `probe_new_backend` | `pending_backend_switch=True` | `backend_probe` |
| `run_validated_backend_experiment` | `post_probe_continuation_required=True` | per-backend task |

## Backend-to-Task Mapping

| Validated Backend | Continuation Task Type |
|---|---|
| `deeplens_geolens_geometric` | `native_lens_simulation_codesign` |
| `deeplens_fresnel_component` | `component_optimization` |
| `phase_to_fft_proxy` | `stable_lens_hsi_codesign` |
| `deeplens_coherent_asm` | `lightweight_psf_probe` |
| other | `lightweight_psf_probe` |
