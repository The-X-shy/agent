# Remote-Aware Handler Routing

Remote-aware routing chooses `remote_native_geolens_validation` only when remote execution is explicitly enabled.

## Modes

| Mode | Remote handler behavior |
|---|---|
| `dry_run` | shown only as a candidate |
| `local` | skipped because the handler requires WSL |
| `remote_opt_in` with `--allow-remote` | selected and executed against the configured worker |
| `remote_opt_in` without `--allow-remote` | stopped before remote execution |

## Phase 44 Behavior

`AgentPlanExecutionLoop` now injects the remote validation design in `remote_opt_in` mode and the evaluator prioritizes remote-required handlers when remote execution is allowed.

Before execution, the selected handler must pass `validate_remote_worker_requirements()`.
