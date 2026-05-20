# Optical-Sensitive HSI Reconstruction

Phase 10 adds optical encoder sensitivity to the HSI reconstruction benchmark.

## What changed from Phase 9

In Phase 9, all 5 encoder types produced identical reconstruction metrics because the forward model summed all spectral bands into a single grayscale measurement and the linear reconstructor only learned per-band scalar multipliers.

Phase 10 introduces:

1. **Enhanced dataset**: `mixed_materials` pattern (default) with K material spectral signatures and spatial abundance mixing. Previous `smooth_low_rank` preserved for backward compat.

2. **Optical-sensitive forward model**: `depth_spectral_coded` mode (default) uses PSF-derived optical features (band spread, centroid, high-frequency energy) to create encoder-specific measurement encoding. Previous `simple_sum` mode preserved for backward compat.

3. **OpticalFeatureExtractor**: Extracts band-dependent features (spread, centroid, high-freq energy) and encoder-level scores (depth stability, spectral separability, coding strength) from PSF cubes.

4. **OpticalConditionedLinearReconstructor**: Uses optical features to create band-dependent spatial basis functions. Higher spectral separability → better per-band recovery.

5. **TinyCNNReconstructor**: Minimal 3-layer CNN. Requires PyTorch; returns `TORCH_NOT_AVAILABLE` when torch is absent.

Phase 11 formalizes TinyCNN as an optional baseline, adds `UNetTinyReconstructor`, and lets Torch-based models receive optical scalar feature maps with `concat_scalar_maps`.

## Important caveats

- **Synthetic only**: All data is synthetic; no real HSI datasets are used.
- **Mock backend only**: PSF cubes come from the mock DeepLens backend; not physically validated.
- **Forward model is an evaluation proxy**: The depth_spectral_coded mode is not a physical sensor model.
- **Linear reconstructor is a baseline**: The optical-conditioned linear reconstructor is not a final reconstruction network.
- **Rankings are for system verification**: Encoder rankings demonstrate the benchmark works, not real HSI performance.
- **Public/local datasets are still scoped**: A local CAVE/ICVL/custom dataset with `mock_deeplens` is not a real camera experiment.
- **DeepLens adapter_proxy is scoped**: Adapter-proxy DeepLens output is not native physical validation.

## Commands

```bash
# Run with optical-sensitive defaults
python -m optiresearch.cli run-hsi-reconstruction \
  --backend mock_deeplens \
  --encoder controlled_chromatic_edof \
  --forward-mode depth_spectral_coded \
  --reconstructor optical_conditioned_linear \
  --dataset-pattern mixed_materials \
  --objective "Evaluate optical-sensitive synthetic HSI reconstruction"

# Run baselines across all 5 encoders
python -m optiresearch.cli run-hsi-baselines \
  --backend mock_deeplens \
  --forward-mode depth_spectral_coded \
  --reconstructor optical_conditioned_linear \
  --dataset-pattern mixed_materials

# Export report
python -m optiresearch.cli export-phase10-report
python -m optiresearch.cli run-hsi-matrix \
  --datasets synthetic \
  --backends mock_deeplens \
  --reconstructors optical_conditioned_linear,tiny_cnn \
  --forward-modes depth_spectral_coded \
  --objective "Compare encoder ranking across reconstructors"
```

## Backward compatibility

Old Phase 9 behavior can be reproduced with:

```bash
python -m optiresearch.cli run-hsi-reconstruction \
  --backend mock_deeplens \
  --encoder controlled_chromatic_edof \
  --forward-mode simple_sum \
  --reconstructor linear_baseline \
  --dataset-pattern smooth_low_rank \
  --objective "..."
```

## Key source files

- `optiresearch/hsi/optical_features.py` — OpticalFeatureExtractor
- `optiresearch/hsi/dataset.py` — SyntheticHSIDataset with multiple patterns
- `optiresearch/hsi/forward_model.py` — HSIForwardModel with 4 modes
- `optiresearch/hsi/reconstruction.py` — OpticalConditionedLinearReconstructor, TinyCNNReconstructor
- `optiresearch/runtime/hsi_pipeline.py` — Full pipeline integration
- `optiresearch/runtime/hsi_baselines.py` — Encoder baseline comparison
- `optiresearch/hsi/public_datasets.py` — Dataset adapters
- `optiresearch/runtime/hsi_matrix.py` — Dataset/reconstructor matrix
