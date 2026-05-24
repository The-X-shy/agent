# Remote Handler End-to-End Validation

Phase 44 validates the full remote-aware handler path:

```text
AgentPlanExecutionLoop
  -> RemoteWorkerRegistry requirement check
  -> SkillRuntimeV2 remote_execution
  -> WSL allowlisted CLI job
  -> RemoteHandlerResult ingestion
  -> ClaimGate / ClaimCeilingResolver
  -> Memory / StateStore
  -> EventBus
  -> AgentPlanExecutionReport
```

## Local Command

```bash
python -m optiresearch.cli run-agent-plan-execution \
  --objective "validate native GeoLens HSI path on WSL through remote-aware handler" \
  --mode remote_opt_in \
  --allow-remote \
  --remote-worker-id windows_wsl \
  --execute-top-k 1
```

## Success Conditions

The successful remote path requires:

- `selected_handler=remote_native_geolens_validation`;
- `execution_target=remote_wsl`;
- `remote_job_id` present;
- `remote_validation_passed=true`;
- `proxy_fallback_used=false`;
- `deeplens_native_psf_path=geolens.psf_geometric`;
- `phase_to_fft_proxy_used=false`;
- returned artifacts under `workspace/remote_jobs/<job_id>`;
- final claim ceiling `native_lens_simulation`.

## Failure Conditions

If WSL execution fails, the system records a structured failure. It does not create a native claim from a failed or fallback result.

The final claim ceiling remains `needs_followup` when remote validation does not pass.
