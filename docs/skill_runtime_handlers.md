# Skill Runtime Handlers

`SkillRuntimeV2` dispatches built-in skills through structured handlers.

## Report Generation

`report_generation` now supports:

- `system_subunit`
- `negative_result`
- `agent_plan_negative_result`

Successful report generation returns `outcome=report_only`.

Unsupported report types return `status=unsupported` and `outcome=structured_unsupported`.

## Plan Execution Handlers

The local plan execution loop handles generated designs as follows:

- `objective_redesign_simpler_metric_mse_only`: local native GeoLens attempt; if DeepLens is unavailable, returns `structured_unsupported`.
- `param_reduction_sweep`: returns `structured_unsupported` until a dedicated reduced-parameter sweep handler exists.
- `alt_param_diffractive_sweep`: local native GeoLens attempt; if DiffractiveLens or DeepLens is unavailable, returns `structured_unsupported`.
- `report_negative_result_doc`: always executable through `report_generation`, returns `report_only`.
- `backend_switch_waveoptics_coherent`: probe-only outcome; returns `needs_followup` when coherent ASM gradients are unavailable.

No handler silently falls back to a proxy backend when the selected design requires native DeepLens evidence.
