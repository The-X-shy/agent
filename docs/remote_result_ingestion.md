# Remote Result Ingestion

Phase 44 defines a strict result contract for `remote_native_geolens_validation`.

## Contract

Remote handler output is parsed into `RemoteHandlerResult`.

Required fields:

| Field | Required value for native validation |
|---|---|
| `execution_target` | `remote_wsl` |
| `status` | `succeeded` |
| `evidence_level` | `native_lens_simulation`, `native_lens_hsi_codesign`, or `rollback_protected_native_lens_hsi` |
| `execution_fidelity` | native DeepLens fidelity, normally `deeplens_native_geometric` |
| `proxy_fallback_used` | `false` |
| `deeplens_native_psf_path` | `geolens.psf_geometric` |
| `full_wave_optics` | `false` |
| `phase_to_fft_proxy_used` | `false` |

Missing required fields produce structured errors. Boolean `false` is preserved and is not treated as missing.

## Failure Rules

A remote result is not accepted as native evidence when:

- the remote job failed;
- any required field is missing;
- a proxy fallback was used;
- `phase_to_fft_proxy_used=true`;
- `deeplens_native_psf_path` is not `geolens.psf_geometric`;
- evidence level is outside the native remote allowlist.

In those cases `remote_validation_passed=false`, `status=failed`, and `evidence_level=needs_followup`.

## Inputs

`parse_remote_handler_result()` can parse:

- a `RemoteJobResult`;
- a dict containing a `RemoteJobResult`;
- `remote_job_result.json`;
- local remote job directories that include `metrics_summary.json`, `result.json`, or artifact manifests.
