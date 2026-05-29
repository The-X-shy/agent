# Component Surrogate HSI Co-design

This handler wires a component surrogate PSF into a synthetic HSI
reconstruction loop:

1. Build synthetic HSI data.
2. Build a differentiable component surrogate PSF.
3. Convolve each HSI band with its PSF.
4. Integrate bands into a measurement.
5. Reconstruct the HSI cube with a lightweight differentiable reconstructor.
6. Backpropagate reconstruction loss to component parameters.
7. Update component parameters and record metrics.

## Outputs

Each run writes:

- `result.json`
- `metrics.json`
- `loss_trace.json`
- `component_parameter_trace.json`
- `psf_artifact.npz`
- `artifact_manifest.json`
- `report.md`

## Evidence Level

Successful runs use:

- `evidence_level=component_surrogate_hsi_codesign`
- `claim_ceiling=component_surrogate_hsi_codesign`

This supports component-level surrogate HSI claims only.
