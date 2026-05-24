# Remote Agent Plan Execution

`run-agent-plan-execution` supports a remote opt-in path for the native GeoLens validation handler.

## Command

```bash
python -m optiresearch.cli run-agent-plan-execution \
  --objective "validate native GeoLens HSI path on WSL through remote-aware handler" \
  --mode remote_opt_in \
  --allow-remote \
  --remote-worker-id windows_wsl \
  --execute-top-k 1
```

## Execution Result Fields

Remote execution records:

| Field |
|---|
| `execution_target` |
| `remote_worker_id` |
| `remote_job_id` |
| `remote_validation_passed` |
| `run_id` |
| `evidence_level` |
| `execution_fidelity` |
| `proxy_fallback_used` |
| `deeplens_native_psf_path` |
| `full_wave_optics` |
| `phase_to_fft_proxy_used` |
| `artifact_return_path` |

The result is then passed to ClaimGate, ResearchMemoryV2, StateStore, EventBus, and the plan execution report.
