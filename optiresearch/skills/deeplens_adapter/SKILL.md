# DeepLens Adapter

Use this skill when a task asks for DeepLens, PSF, MTF, differentiable optics, optical simulation, EDOF, HSI encoder simulation, or mock optical backend execution.

MVP execution uses `MockDeepLensAdapter`, not real DeepLens. The output is valid for software flow testing and claim-evidence wiring, but must be described as mock-backed simulation only.

## Inputs

- `optical_spec`: optical design parameters such as `psf_size`, target encoder type, and backend.
- `sweep_spec`: sweep parameters such as `depth_planes` and `wavelength_bands`.
- `seed`: deterministic seed, default `42`.

## Outputs

- `psf_cube.npz`
- `mtf_curves.csv`
- `optical_metrics.json`
- `run_manifest.json`
- `psf_grid.png` when matplotlib is available

## Flow

1. Build a structured optical spec.
2. Run `run_mock_psf` through the allowlisted skill executor.
3. Register every produced file in ArtifactStore.
4. Write Meta-Trace with artifact IDs and metric findings.
5. Treat claims as simulation-only unless real DeepLens or prototype evidence is attached.
