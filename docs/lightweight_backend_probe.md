# Lightweight Backend Probe

## Purpose

A fast (<5s), dependency-light validation run to confirm a backend is
available and functional before running full experiments on it.

## Function

`run_lightweight_backend_probe(backend_id, device="cpu")` in
`optiresearch/runtime/lightweight_experiments.py`

## Behavior by Backend Type

### Phase-to-FFT Proxy / Mock / Synthetic
- Always succeeds (no external dependencies)
- Runs minimal FFT PSF generation (2 bands, 7x7 PSF)
- Returns `differentiable=True`, `gradient_norm=None`

### DeepLens Backends (deeplens_geolens_geometric, etc.)
- First checks `deeplens.geolens` importability
- If DeepLens unavailable: returns `status=failed` with
  `error_code=DEEPLENS_UNAVAILABLE` and `claim_gate_decision=needs_followup`
- If DeepLens available: runs FFT PSF probe for basic validation
- Returns `deeplens_available=True`

## Output

Returns a `ControllerResult` with `result_payload`:
- `backend_available`: bool
- `deeplens_available`: bool
- `probe_time_seconds`: float
- `probe_method`: "fft_fraunhofer"
- `probe_status`: "succeeded" | "unavailable" | error message
- `psf_width_x`, `psf_width_y`, `psf_energy`: float
- `differentiable`: bool

## Probe Depths

### Shallow (default)
- FFT-based Fraunhofer diffraction PSF
- Fast (<5s), no external dependencies
- Validates basic differentiability and PSF metrics

### Deep (Phase 32)
- Actually calls `deeplens.geolens.psf(points, wvln, ks, model="geometric")`
- Loads real lens file (cooke.json)
- Backpropagates through PSF to verify `requires_grad=True`
- Reports `optical_gradient_norm`, `parameters_changed`
- Evidence level: `native_lens_simulation`

## CLI

```bash
# Shallow probe (FFT-based)
python -m optiresearch.cli run-lightweight-backend-probe \
  --backend-id deeplens_geolens_geometric

# Deep probe (actual GeoLens geometric PSF)
python -m optiresearch.cli run-lightweight-backend-probe \
  --backend-id deeplens_geolens_geometric \
  --probe-depth deep
```

## Safety

- Never crashes — returns structured failure on any exception
- Runs on CPU only
- No file system changes
- No remote execution
