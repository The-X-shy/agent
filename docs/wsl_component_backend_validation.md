# WSL Component Backend Validation

Phase 62 runs real component-level validation on a Windows WSL worker.

## Worker

- **Worker ID:** `windows_wsl`
- **Connection:** SSH via `SSHRemoteRunner`
- **Python:** `/mnt/d/agent/run_agent_python.sh`

## Validation Sequence

```bash
# Step 1: Discovery
python -m optiresearch.cli run-remote-discover-deeplens-components \
    --worker-id windows_wsl

# Step 2: Fresnel probe
python -m optiresearch.cli run-remote-deeplens-component-probe \
    --worker-id windows_wsl --component fresnel --device cpu

# Step 3: Binary2Phase probe
python -m optiresearch.cli run-remote-deeplens-component-probe \
    --worker-id windows_wsl --component binary2phase --device cpu

# Step 4: Diffractive candidate probe
python -m optiresearch.cli run-remote-deeplens-component-probe \
    --worker-id windows_wsl --component diffractive --device cpu

# Step 5: Agent plan execution pivot
python -m optiresearch.cli run-agent-plan-execution \
    --objective "validate component-level DeepLens optimization after GeoLens autograd failure" \
    --seed-result-path workspace/remote_jobs/remote_job_53f5e98e37bdeed0/result.json \
    --mode remote_opt_in \
    --use-gradient-diagnosis \
    --allow-remote \
    --remote-worker-id windows_wsl \
    --execute-top-k 2
```

## Expected Outcomes

- At least one of Fresnel or Binary2Phase achieves `native_component_optimization`
- OR all components return structured `needs_followup` with `error_code=DEEPLENS_COMPONENT_API_UNAVAILABLE`
- `full_geolens_direct_update` is NOT selected by agent loop
- Claim gate caps component claims at `native_component_optimization`
