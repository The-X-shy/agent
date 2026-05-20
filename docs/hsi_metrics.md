# HSI Metrics

Phase 9 reports:

- `PSNR`
- `SSIM`
- `SAM`
- `ERGAS`
- `per_band_RMSE`
- `worst_depth_SAM`

Phase 10 adds optical feature metrics:

- `coding_strength`
- `depth_stability_score`
- `spectral_separability_score`
- `band_condition_score`

All metrics are implemented in numpy and are JSON serializable.

Reconstruction ranking uses `reconstruction_score = PSNR - 5.0*SAM - 0.02*ERGAS`.

Evidence policy:

- optical-only claims use optical metrics;
- reconstruction-level claims require `reconstruction_metrics.json`;
- mock HSI results must remain mock-backed;
- DeepLens proxy or semi-native HSI results must not be described as native physical validation;
- synthetic HSI reconstruction rankings are for system verification only, not real-world conclusions.
