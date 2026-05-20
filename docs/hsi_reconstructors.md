# HSI Reconstructors

Phase 11 supports these reconstructors:

| Reconstructor | Dependency | Default test dependency | Notes |
|---|---|---:|---|
| `linear_baseline` | numpy | yes | simple scalar per-band baseline |
| `optical_conditioned_linear` | numpy | yes | default optical-sensitive baseline |
| `tiny_cnn` | torch | no | optional small CNN baseline |
| `unet_tiny` | torch | no | optional interface for stronger networks |

If Torch is unavailable, `tiny_cnn` and `unet_tiny` return `TORCH_NOT_AVAILABLE` and the matrix marks those rows as skipped.

## Optical Feature Injection

`concat_scalar_maps` expands these scalar optical features into constant `H x W` maps and concatenates them with the measurement:

- `spectral_separability_score`
- `depth_stability_score`
- `coding_strength`
- `band_condition_score`

`conditioning_vector` is reserved in manifests for future models.

## Commands

```bash
python -m optiresearch.cli run-hsi-reconstruction \
  --dataset synthetic \
  --backend mock_deeplens \
  --encoder controlled_chromatic_edof \
  --forward-mode depth_spectral_coded \
  --reconstructor tiny_cnn \
  --tiny-cnn-epochs 2 \
  --use-optical-feature-maps \
  --objective "Evaluate synthetic HSI reconstruction with tiny CNN"
```

TinyCNN/UNet results are optional baselines, not final paper-scale networks.

