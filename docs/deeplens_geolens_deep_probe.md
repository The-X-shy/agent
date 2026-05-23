# DeepLens GeoLens Geometric Deep Probe

## Purpose

The deep probe actually exercises the DeepLens GeoLens geometric PSF path,
unlike the shallow probe which uses FFT-based proxy PSF generation.

## Function

`run_deeplens_geolens_geometric_deep_probe(backend_id, device="cpu")` in
`optiresearch/runtime/lightweight_experiments.py`

## Behavior

1. Checks `deeplens.geolens` importability
2. Searches for `cooke.json` lens file in known locations
3. Instantiates `GeoLens(lens_path, device=device)`
4. Calls `geolens.psf(points, wvln, ks, model="geometric")`
5. Backpropagates through PSF to verify `requires_grad=True`
6. Reports gradient norm and parameter changes

## Output Fields

| Field | Type | Meaning |
|---|---|---|
| `differentiable` | bool | `psf.requires_grad` was True |
| `optical_gradient_norm` | float | Max gradient norm across parameters |
| `parameters_changed` | bool | `gradient_norm > 0` |
| `deeplens_native_psf_path` | str | `geolens.psf_geometric` |
| `full_wave_optics` | bool | Always `False` for geometric path |
| `phase_to_fft_proxy_used` | bool | Always `False` for deep probe |
| `evidence_level` | str | `native_lens_simulation` |

## Failure Modes

| Error Code | Meaning |
|---|---|
| `DEEPLENS_UNAVAILABLE` | `deeplens.geolens` not importable |
| `LENS_FILE_NOT_FOUND` | `cooke.json` not found in known paths |
| `GEOLENS_PSF_GEOMETRIC_FAILED_*` | Error during PSF generation or backward |

## CLI

```bash
python -m optiresearch.cli run-lightweight-backend-probe \
  --backend-id deeplens_geolens_geometric \
  --probe-depth deep
```

## Opt-in Testing

```bash
OPTIRESEARCH_ENABLE_REAL_DEEPLENS_TESTS=1 \
python -m pytest tests/test_real_deeplens_geolens_deep_probe.py
```
