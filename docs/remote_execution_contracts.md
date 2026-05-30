# Remote Execution Contracts

Remote execution contracts govern WSL worker command execution safety and artifact return policies.

## Contract Fields

| Field | Description |
|---|---|
| command_name | CLI command to execute remotely |
| handler_id | Associated handler |
| allowed_args | Whitelisted command arguments |
| forbidden_args | Explicitly blocked arguments |
| required_worker_capabilities | Worker tags required (e.g., deeplens_available) |
| timeout_sec | Maximum execution time |
| output_dir_policy | required / optional / none |
| artifact_return_policy | required / optional / none |
| allowlist_entry_required | Whether command must be in allowlist |
| remote_job_id_required | Whether job tracking ID is required |

## Core Remote Contracts

8 remote execution contracts are defined:

1. `rec_trainable_param_inspection`
2. `rec_autograd_audit`
3. `rec_component_first_probe`
4. `rec_stabilized_native_geolens_hsi`
5. `rec_native_geolens_benchmark`
6. `rec_benchmark_failure_analysis`
7. `rec_resume_benchmark`
8. `rec_component_surrogate_hsi_codesign`

## Validation

```bash
python -m optiresearch.cli validate-remote-execution-contracts
```

Safety checks:
- Command is in the remote command allowlist
- Allowed args do not overlap with forbidden args
- No shell metacharacters in args (; | $())
- Timeouts are within reasonable bounds
- Result parsers are specified
