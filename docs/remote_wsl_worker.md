# Remote WSL Worker

Phase 17 adds a Windows WSL worker path for real DeepLens source execution. The Mac process remains the controller, memory store, evidence gate, and report writer. The WSL process runs allowed OptiResearch CLI jobs only.

Phase 24 adds `ExperimentControllerV2.run_remote()` which delegates to the remote worker
system with backend capability validation and claim ceiling enforcement.

See [experiment_controller_v2.md](experiment_controller_v2.md) for the unified experiment entry point.

## Worker Layout

```text
WSL project: /mnt/d/agent
WSL DeepLens source: /mnt/d/external/DeepLens
Python wrapper: /mnt/d/agent/run_agent_python.sh
Required env: DEEPLENS_REPO_PATH=/mnt/d/external/DeepLens
```

`run_agent_python.sh` should run the project virtualenv Python after exporting `DEEPLENS_REPO_PATH`.

## Register Worker

```bash
python -m optiresearch.cli add-remote-worker \
  --worker-id windows_wsl \
  --host wslbox \
  --port 22 \
  --username ysl \
  --remote-project-dir /mnt/d/agent \
  --remote-workspace-dir /mnt/d/agent/workspace \
  --python-executable /mnt/d/agent/run_agent_python.sh
```

The registry is stored at:

```text
workspace/remote_workers/workers.json
```

## Allowed Remote Commands

Remote execution accepts argument lists only. It rejects shell fragments such as `;`, `&&`, `||`, pipes, redirection, backticks, command substitution, `sudo`, `rm -rf`, `chmod 777`, and `python -c`.

Allowed OptiResearch commands:

- `check-deeplens`
- `probe-deeplens-source`
- `inspect-deeplens-source`
- `run-deeplens-source-smoke`
- `run-hsi-reconstruction`
- `run-hsi-matrix`
- `run-codesign-loop`
- `run-autonomous-loop`

## Run Remote Jobs

```bash
python -m optiresearch.cli check-remote-worker --worker-id windows_wsl

python -m optiresearch.cli run-remote-deeplens-source-smoke \
  --worker-id windows_wsl

python -m optiresearch.cli run-remote-codesign \
  --worker-id windows_wsl \
  --objective "Run strict DeepLens-backed co-design on WSL D drive worker" \
  --psf-source deeplens_parameterized \
  --backend deeplens \
  --fallback-policy fail \
  --max-iterations 2
```

Remote output is pulled back to:

```text
workspace/remote_jobs/<job_id>/
```

The local controller ingests `artifact_manifest.json` and `metrics_summary.json`, then writes ArtifactStore, RunMemory, and ClaimEvidence entries.

## Phase 19: Native Optimization Probe (Remote)

Run native differentiable optimization probes on the WSL worker:

```bash
# Remote inspection
python -m optiresearch.cli inspect-deeplens-native-optimization

# Remote native probe
python -m optiresearch.cli run-remote-native-optimization-probe \
  --worker-id windows_wsl \
  --lens-class ParaxialLens \
  --objective minimize_psf_width \
  --max-steps 2 \
  --device cpu

# Export remote execution report
python -m optiresearch.cli export-remote-execution-report --job-id <job_id>
```

See `docs/deeplens_native_optimization_probe.md` for details.

## Evidence Boundary

Successful DeepLens source smoke can support a DeepLens-backed smoke claim. Successful strict co-design without fallback can support a DeepLens-backed black-box co-design claim.

It still cannot support native differentiable optimization, native DeepLens parameter update, real camera validation, or real HSI optical performance claims.
