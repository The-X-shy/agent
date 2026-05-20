# ExperimentSpec v0.1

`ExperimentSpec` v0.1 is the frozen exchange format between planning, adapter execution, memory, and paper experiment reporting.

All four spec objects include:

- `schema_version`: fixed string, default `0.1`.

Use `validate_experiment_spec_version(spec)` before adapter execution. A version mismatch is treated as schema drift and must fail clearly.

## OpticalSpec

| Field | Unit | Default In Mock EDOF-HSI | Notes |
|---|---:|---|---|
| `schema_version` | none | `0.1` | Frozen v0.1 marker |
| `spec_id` | none | deterministic id | Stable identifier |
| `encoder_type` | none | `controlled_chromatic_edof` | One of `conventional`, `achromatic`, `edof`, `chromatic_coded`, `controlled_chromatic_edof`, `mock` |
| `aperture` | mm | `None` | Optional until real lens import |
| `focal_length` | mm | `None` | Optional until real lens import |
| `f_number` | none | `2.8` | Optical f-number |
| `sensor_type` | none | `hsi` | One of `mono`, `rgb`, `hsi`, `mock` |
| `wavelength_range_nm` | nm | `(450.0, 700.0)` | Inclusive wavelength range |
| `wavelength_bands` | count | `31` | Must be >= 1 |
| `depth_range_mm` | mm | `(-4.0, 4.0)` | Inclusive depth range |
| `depth_planes` | count | `9` | Must be >= 1 |
| `psf_size` | pixels | `32` | Square PSF side length |
| `constraints` | none | target flags | Extensible constraints |
| `metadata` | none | backend and encoder type | Extensible metadata |

## SweepSpec

| Field | Unit | Default | Notes |
|---|---:|---|---|
| `schema_version` | none | `0.1` | Frozen v0.1 marker |
| `sweep_id` | none | deterministic id | Stable identifier |
| `wavelengths_nm` | nm | 31 uniform samples | Explicit wavelength sweep |
| `depths_mm` | mm | 9 uniform samples | Explicit depth sweep |
| `fields` | normalized field | `[0.0]` | Field positions |
| `seeds` | none | `[42]` | Deterministic mock seeds |
| `metadata` | none | sampling info | Extensible metadata |

## MetricSpec

| Field | Unit | Default | Notes |
|---|---:|---|---|
| `schema_version` | none | `0.1` | Frozen v0.1 marker |
| `metric_id` | none | deterministic id | Stable identifier |
| `optical_metrics` | none | depth, spectral, MTF, efficiency | Backend metrics |
| `reconstruction_metrics` | none | `[]` | Reserved for reconstruction tasks |
| `evidence_metrics` | none | artifact and trace support | Evidence audit metrics |
| `primary_metric` | none | `psf_depth_similarity` | Primary optimization metric |
| `maximize` | none | `True` | Direction of primary metric |
| `thresholds` | none | depth >= 0.8, spectral >= 0.3 | Evidence thresholds |
| `metadata` | none | evidence policy | Extensible metadata |

## ExperimentSpec

| Field | Unit | Default | Notes |
|---|---:|---|---|
| `schema_version` | none | `0.1` | Frozen v0.1 marker |
| `experiment_id` | none | deterministic id | Stable identifier |
| `objective` | text | user objective | Natural-language research goal |
| `optical_spec` | object | `OpticalSpec` | Optical configuration |
| `sweep_spec` | object | `SweepSpec` | Sweep grid |
| `metric_spec` | object | `MetricSpec` | Metrics and thresholds |
| `backend` | none | `mock_deeplens` | One of `mock_deeplens`, `deeplens`, `metasurface_mock` |
| `run_budget` | none | one run, 60 seconds | Extensible budget metadata |
| `created_by` | none | `MethodBuilder` | Producer role |
| `metadata` | none | kind and encoder type | Extensible metadata |

## Extensible Fields

Only these fields are intended for extension without changing v0.1:

- `constraints`
- `metadata`
- `run_budget`
- metric lists
- `thresholds`

## Forbidden v0.1 Changes

Within schema version `0.1`, do not rename, remove, or change units for any required field. Do not change enum values silently. Additive backend-specific values require a new schema version or a compatibility layer.
