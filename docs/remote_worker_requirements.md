# Remote Worker Requirements

Phase 44 validates remote handler requirements before running a WSL job.

## Validation Function

`validate_remote_worker_requirements(handler_capability, worker_id)` checks the registered worker against the selected handler.

For `remote_native_geolens_validation`, the required worker tags are:

| Requirement |
|---|
| `windows_wsl` |
| `deeplens_available` |
| `geolens_psf_geometric` |

The worker also must pass:

| Check | Meaning |
|---|---|
| Worker exists | `worker_id` is present in `RemoteWorkerRegistry` |
| Command allowlist | the generated remote command is accepted by `validate_remote_command()` |
| Worker command scope | `capabilities.allowed_commands` includes `run-deeplens-native-geolens-hsi-codesign`, when the list is present |
| Runtime | `max_runtime_seconds` is sufficient for the handler timeout |
| Artifact path | returned artifacts stay under `remote_workspace_dir/remote_jobs` |

## Failure Behavior

If validation fails:

- plan execution stops before remote command execution;
- `status=stopped`;
- `stop_reason=remote_worker_requirements_not_met`;
- `remote_validation_passed=false`;
- final claim ceiling remains `needs_followup`.

No arbitrary shell command is constructed or executed by this validation path.
