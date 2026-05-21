# DeepLens Native Differentiable Optimization Probe

## Overview

The native optimization probe systematically tests whether DeepLens lens classes
support true gradient-based optical optimization. Unlike the black-box parameterized
PSF approach (Phase 16-18), this probe executes the full autograd chain:

```
optical parameter → PSF simulation → scalar loss → backward → optimizer.step → parameter change
```

## Probe Architecture

### Inspector (`inspect-deeplens-native-optimization`)

Scans each DeepLens lens class at runtime to detect:
- `activate_grad` — enables gradient tracking
- `get_optimizer` — returns a torch optimizer
- `get_optimizer_params` / `parameters()` — trainable parameters
- `psf()` / `render()` / `forward()` — PSF generation methods
- Constructor requirements (foclen, fnum vs lens file)

### Probe Runner (`run-native-optimization-probe`)

For a given lens class and objective, executes the full differentiable loop:
1. Import and instantiate the lens
2. Call `activate_grad(True)`
3. Generate PSF (keeping gradients)
4. Compute scalar loss (PSF width / center intensity / target match)
5. `loss.backward()` — check gradient norm
6. `optimizer.step()` — check parameter change
7. Recompute PSF and loss — compare before/after

## Lens Classes

| Class | Constructor | Differentiable? | Notes |
|---|---|---|---|
| ParaxialLens | `(foclen, fnum, device)` | Likely yes | No lens file needed; primary probe target |
| GeoLens | `(filename, device)` | Unknown | Requires JSON lens file |
| DiffractiveLens | `(filename, device)` | Unknown | Requires JSON lens file |
| HybridLens | `(filename, device)` | Unknown | Requires JSON lens file |
| PSFNetLens | `(lens_path, ...)` | Unknown | Neural surrogate; requires file |

## Realization Levels

- **native** — Full autograd chain works: `gradient_norm > 0` AND `parameters_changed = True`
- **semi_native** — `activate_grad` works but optimizer/parameter change fails
- **adapter_proxy** — Uses existing DeepLensAdapter (non-differentiable)
- **unavailable** — DeepLens not installed

## Objectives

- `minimize_psf_width` — Minimize spatial variance of PSF (centroid-weighted second moment)
- `maximize_center_intensity` — Maximize peak PSF intensity
- `match_target_psf` — MSE against a Gaussian target PSF
- `hsi_reconstruction_loss` — Future: HSI reconstruction quality

## Usage

```bash
# Inspect native optimization capabilities
python -m optiresearch.cli inspect-deeplens-native-optimization

# Run a minimal native probe on ParaxialLens
python -m optiresearch.cli run-native-optimization-probe \
  --lens-class ParaxialLens \
  --objective minimize_psf_width \
  --max-steps 2 \
  --device cpu

# Run remotely on WSL worker
python -m optiresearch.cli run-remote-native-optimization-probe \
  --worker-id windows_wsl \
  --lens-class ParaxialLens \
  --objective minimize_psf_width \
  --max-steps 2 \
  --device cpu

# Export report
python -m optiresearch.cli export-phase19-report
```

## Constraints

- No silent fallback to mock optimization
- No claiming native optimization without actual `backward()` + `optimizer.step()` verification
- `gradient_norm > 0` and `parameters_changed = True` are required for native classification
- Black-box search is explicitly not differentiable optimization
- Phase 19B separates component, lens, and optical-HSI claims. A surface-level success does not prove full native optical-HSI co-design.

## Phase 19B Surface and Lens-file Probes

```bash
python -m optiresearch.cli scan-deeplens-optimization-paths

python -m optiresearch.cli run-deeplens-surface-optimization-probe \
  --surface Fresnel \
  --objective minimize_phase_variance \
  --max-steps 3

python -m optiresearch.cli run-deeplens-surface-optimization-probe \
  --surface Binary2Phase \
  --objective match_target_phase \
  --max-steps 3

python -m optiresearch.cli run-deeplens-lensfile-optimization-probe \
  --lens-class GeoLens \
  --max-files 5 \
  --max-steps 2

python -m optiresearch.cli export-phase19b-report
```

## Artifacts

Each probe produces:
```
workspace/native_optimization/<probe_id>/
├── probe_spec.json
├── probe_result.json
├── loss_trace.json
├── parameter_snapshot.json
├── psf_before.npz
├── psf_after.npz
└── native_probe_report.md
```

Surface probes write:

```
workspace/native_optimization/surface_probe_<id>/
├── probe_spec.json
├── probe_result.json
├── loss_trace.json
├── parameter_before.json
├── parameter_after.json
├── phase_before.npz
├── phase_after.npz
└── report.md
```
