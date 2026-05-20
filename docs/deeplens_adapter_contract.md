# DeepLens Adapter Contract

The real DeepLens integration must implement the same result contract as the mock backend. Importing `DeepLensAdapter` must never require DeepLens to be installed.

## Shared Result Types

Defined in `optiresearch/adapters/base.py`:

- `AdapterArtifact`
- `AdapterMetricBundle`
- `AdapterRunResult`
- `OpticalAdapterProtocol`

`AdapterRunResult` is the only execution result format for mock and real backends.

Required top-level fields:

| Field | Meaning |
|---|---|
| `status` | `succeeded`, `failed`, or `skipped` |
| `artifacts` | Produced artifact file paths before registration |
| `artifact_refs` | `AdapterArtifact` descriptors |
| `metric_bundle` | `AdapterMetricBundle` |
| `logs` | Human-readable execution logs |
| `errors` | Structured error dictionaries |
| `metadata` | Backend and operation metadata |

For backward compatibility, `result["metrics"]` and `result.metrics` return `result.metric_bundle.metrics`.

## DeepLensAdapter Methods

| Method | Contract |
|---|---|
| `validate_environment()` | Returns environment details without raising when DeepLens is missing |
| `translate_experiment_spec()` | Accepts frozen `ExperimentSpec` v0.1 or dict and returns `DeepLensCandidateConfig` |
| `simulate_psf_cube()` | Returns `AdapterRunResult` for PSF simulation |
| `compute_mtf()` | Returns `AdapterRunResult` for MTF computation |
| `run_optimization()` | Returns `AdapterRunResult` for optimization |
| `collect_artifacts()` | Converts backend output files into `AdapterArtifact` descriptors |

## Missing Backend Behavior

If `deeplens` is not installed, methods return structured failures:

```json
{
  "code": "DEEPLENS_NOT_INSTALLED",
  "message": "The real deeplens package is not installed.",
  "hint": "Use MockDeepLensAdapter for local tests, or install the project-specific DeepLens backend."
}
```

This preserves local tests, CLI flows, and paper-report generation without a GPU or real DeepLens installation.

## Phase 5 Integration Status

Phase 5 adds a minimal real-backend smoke path while preserving the mock path.

The supported upstream repository is:

- `https://github.com/vccimaging/DeepLens`

DeepLens currently packages as `deeplens-core` and imports as `deeplens`. Its published project metadata requires Python `>=3.12` and includes PyTorch-based dependencies. The local OptiResearch test environment can remain lighter; real DeepLens tests stay opt-in.

`validate_environment()` returns:

- `available`
- `error_code`
- `message`
- `python_version`
- `deeplens_version`
- `import_path`
- `capabilities`
- Phase 4 compatibility fields: `ok`, `backend`, `error`

`translate_experiment_spec()` now produces `DeepLensCandidateConfig` with:

- `wavelengths_nm`
- `depths_mm`
- `psf_size`
- `encoder_type`
- `sensor_type`
- `backend="deeplens"`
- `notes`
- `unsupported_fields`
- `source_spec`

Unsupported `ExperimentSpec v0.1` fields are preserved in `unsupported_fields`. The adapter must not silently discard them.

`simulate_psf_cube()` behavior:

1. Imports `deeplens` if available.
2. Returns `DEEPLENS_NOT_INSTALLED` if unavailable.
3. Preferentially uses `vccimaging/DeepLens` `ParaxialLens.psf(points, ks=...)` for the minimal smoke cube.
4. Falls back to module-level smoke call names if present.
5. Returns `DEEPLENS_API_UNSUPPORTED` when the installed package does not expose a compatible smoke API.
6. On success, writes `psf_cube.npz`, `mtf_curves.csv`, `optical_metrics.json`, and `run_manifest.json`, then returns them as `AdapterArtifact` entries.

Install in a Python 3.12+ environment:

```bash
python -m pip install "deeplens-core @ git+https://github.com/vccimaging/DeepLens.git"
```

Real DeepLens tests are opt-in:

```bash
OPTIRESEARCH_ENABLE_REAL_DEEPLENS_TESTS=1 python -m pytest tests/test_deeplens_environment.py
```

## Phase 6 Capability Model

`validate_environment()` returns a stable capability list. Each capability has:

```json
{
  "name": "import_deeplens",
  "available": true,
  "reason": "deeplens module import succeeded",
  "evidence": "import check"
}
```

Tracked capabilities:

- `import_deeplens`
- `paraxial_lens_available`
- `psf_smoke_available`
- `mtf_export_available`
- `encoder_specific_design_available`
- `optimization_available`
- `hsi_pipeline_available`

Current DeepLens 1.5.2 support status:

- `ParaxialLens.psf(points, ks=...)` is used for PSF smoke validation.
- `mtf_export_available` means the adapter can export a simple MTF CSV from smoke PSF output.
- `encoder_specific_design_available=false`: encoder families are preserved in metadata but not yet mapped to distinct real optical designs.
- `optimization_available=false`: optimization remains a structured contract placeholder.
- `hsi_pipeline_available=false`: the smoke path replicates monochrome PSF behavior across wavelength bands.

## Smoke-Level vs Full Encoder Behavior

Smoke-level DeepLens runs validate:

- DeepLens import and environment detection.
- Adapter execution.
- Standard artifact creation.
- ArtifactStore registration.
- RunMemory compilation.
- ClaimEvidence wiring.

Smoke-level DeepLens runs do not validate:

- Controlled chromatic EDOF superiority.
- Encoder-specific optical tradeoffs.
- Wavelength-dependent HSI behavior.
- Optimization-backed design rules.

Structured error policy remains mandatory: missing packages, API mismatch, and unimplemented optimization must return `AdapterRunResult(status="failed", errors=[...])`, not a traceback.

## Real Backend Binding

The future binding must map `ExperimentSpec v0.1` to DeepLens-native lens, sensor, wavelength, depth, and optimization objects. It must not change `ExperimentSpec v0.1`; compatibility work belongs in `translate_experiment_spec()`.

## Phase 7 Encoder Proxy Contract

Phase 7 upgrades DeepLens from smoke-level to encoder-specific baseline behavior by combining:

1. real DeepLens `ParaxialLens` base PSF generation;
2. deterministic adapter-level encoder proxy transforms;
3. explicit proxy metadata and caveats.

New capabilities:

- `encoder_specific_proxy_available`
- `encoder_specific_native_available`
- `proxy_transform_available`
- `raw_base_psf_export_available`
- `proxy_manifest_export_available`

Current expected status with DeepLens 1.5.2:

- `encoder_specific_proxy_available=true`
- `encoder_specific_native_available=false`
- `proxy_transform_available=true`
- `raw_base_psf_export_available=true`
- `proxy_manifest_export_available=true`

The adapter writes:

- `raw_base_psf_cube.npz`
- `psf_cube.npz`
- `proxy_transform_manifest.json`
- `optical_metrics.json`
- `mtf_curves.csv`
- `run_manifest.json`

Required metadata:

- `encoder_behavior_realized=true`
- `encoder_behavior_realization_level="adapter_proxy"`
- `physical_validation_level="deeplens_base_psf_plus_adapter_proxy"`
- `proxy_transform_applied=true`
- `proxy_transform_name`

This evidence can support adapter-level encoder behavior claims. It cannot support native physical encoder optimization claims.

## Phase 8 Semi-Native Contract

`simulate_psf_cube(..., realization="auto")` supports:

- `auto`
- `adapter_proxy`
- `semi_native`
- `native`

Current behavior:

- `conventional` can select `semi_native` when `ParaxialLens` is available.
- Other encoders remain `adapter_proxy` unless experimental semi-native support is enabled and API probe detects suitable phase/surface classes.
- `native` falls back to proxy until native DeepLens encoder designs are bound.

New artifact:

- `realization_manifest.json`

New metric fields:

- `selected_realization_level`
- `semi_native_attempted`
- `semi_native_succeeded`
- `proxy_fallback_used`
- `claim_scope`

Semi-native evidence is narrower than native validation. It cannot support claims about native physical optimization.
