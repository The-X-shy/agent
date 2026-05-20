# OptimizationSpec 0.1 Draft

`OptimizationSpec` is a draft schema for Phase 9. It is not frozen.

Fields:

- `schema_version="0.1-draft"`
- `optimization_id`
- `target_metrics`
- `loss_terms`
- `variables`
- `constraints`
- `max_iterations`
- `budget`
- `backend`
- `requires_native_support`
- `metadata`

Current DeepLens behavior:

```text
DeepLensAdapter.run_optimization(...) -> OPTIMIZATION_NOT_AVAILABLE
```

The adapter must not pretend that optimization completed before native optimization support is bound.
