# HSI Matrix Protocol

Phase 11 compares encoder ranking across datasets, optical backends, reconstructors, and forward modes.

Default matrix:

- datasets: `synthetic`
- backends: `mock_deeplens`
- encoders: `conventional`, `achromatic`, `edof`, `chromatic_coded`, `controlled_chromatic_edof`
- reconstructors: `optical_conditioned_linear`, plus `tiny_cnn` when Torch is available
- forward modes: `depth_spectral_coded`

Output:

```text
workspace/hsi/matrix/<matrix_id>/
├── hsi_matrix_results.json
├── hsi_matrix_results.md
└── hsi_matrix_summary.json
```

Table fields include dataset, backend, encoder, reconstructor, forward mode, PSNR, SSIM, SAM, ERGAS, worst-depth SAM, reconstruction score, rank, evidence level, and caveat.

## Command

```bash
python -m optiresearch.cli run-hsi-matrix \
  --datasets synthetic \
  --backends mock_deeplens \
  --reconstructors optical_conditioned_linear,tiny_cnn \
  --forward-modes depth_spectral_coded \
  --objective "Compare encoder ranking across reconstructors"
```

## ClaimEvidence Rules

Matrix-level claims must distinguish:

- dataset: synthetic, local_npz, cave, icvl;
- backend: mock_deeplens, deeplens;
- reconstructor: optical_conditioned_linear, tiny_cnn, unet_tiny;
- realization level: mock, adapter_proxy, semi_native, native.

Synthetic/public dataset results with a mock optical encoder are not real camera validation. DeepLens adapter_proxy is not native physical validation.

See `docs/public_hsi_matrix.md` for local/public dataset matrix runs.
