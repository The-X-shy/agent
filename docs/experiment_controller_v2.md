# Experiment Controller v2

`ExperimentControllerV2` is the unified entry point for all differentiable optics
experiments. It wraps Phase 18-23 runtime loops behind a single controller that
validates backend capabilities, enforces claim ceilings, and delegates to the
strategy engine for next-action recommendations.

## Supported Task Types

| task_type | Runtime Loop | Minimum Claim Ceiling |
|---|---|---|
| native_optimization_probe | Phase 19 probe | native_component_optimization |
| native_hsi_codesign | Phase 20 loop | native_hsi_proxy |
| native_hsi_reconstruction_codesign | Phase 21 loop | native_full_reconstruction_proxy |
| native_waveoptics_codesign | Phase 22 loop | native_waveoptics |
| stable_lens_hsi_codesign | Phase 23 loop | native_lens_simulation |
| psf_probe | PSF generation | deeplens_integration_smoke |
| component_optimization | Component-level | native_component_optimization |

## Claim Ceiling Enforcement

When a task requires a claim level above the backend's ceiling, the controller
automatically downgrades the claim:

```python
spec = ExperimentSpecV2(
    spec_id="test",
    task_type="native_waveoptics_codesign",
    backend_id="deeplens_geolens_geometric",  # ceiling: native_lens_simulation
)
result = ctrl.run_local(spec)
# result.status == "claim_downgraded"
# result.downgraded_from == "native_waveoptics"
# result.downgraded_to == "native_lens_simulation"
```

## Workflow

1. `plan_experiment()` — create experiment spec
2. `validate_preconditions()` — check backend supports task
3. `run_local()` / `run_remote()` — execute experiment
4. `evaluate_metrics()` — extract key metrics
5. `update_memory()` — write to research memory
6. `update_claim_evidence()` — register claim
7. `recommend_next_action()` — delegate to strategy engine

## CLI

```bash
python -m optiresearch.cli run-experiment-v2 \
  --backend-id deeplens_geolens_geometric \
  --task-type stable_lens_hsi_codesign \
  --execution-target local
```

## Programmatic API

```python
from optiresearch.runtime.experiment_controller_v2 import (
    ExperimentControllerV2,
    ExperimentSpecV2,
)

ctrl = ExperimentControllerV2()
spec = ctrl.plan_experiment(
    "test stability",
    "deeplens_geolens_geometric",
    "stable_lens_hsi_codesign",
)
result = ctrl.run_local(spec)
```
