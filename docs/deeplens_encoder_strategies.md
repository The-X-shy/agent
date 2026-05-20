# DeepLens Encoder Strategies

Phase 7 defines one strategy for each baseline encoder family.

| Encoder | Realization Level | Strategy | Purpose |
|---|---|---|---|
| conventional | adapter_proxy | conventional_paraxial_depth_proxy | Baseline ParaxialLens PSF with weak depth variation. |
| achromatic | adapter_proxy | achromatic_shared_wavelength_proxy | Reduce wavelength-dependent PSF variation. |
| edof | adapter_proxy | edof_depth_smoothing_proxy | Increase depth invariance through depth smoothing. |
| chromatic_coded | adapter_proxy | chromatic_spatial_modulation_proxy | Increase wavelength separability through spatial modulation. |
| controlled_chromatic_edof | adapter_proxy | controlled_chromatic_edof_joint_proxy | Combine depth smoothing and controlled wavelength coding. |

Current Phase 8 backend is:

- real DeepLens base PSF generation;
- `semi_native` conventional ParaxialLens baseline when supported;
- adapter-level encoder proxy transformation for encoders without lens-side support;
- not native physical encoder optimization;
- suitable for testing system pipeline and preliminary encoder-specific protocol;
- not sufficient for final optical performance claims.

`semi_native` means part of the behavior uses DeepLens native lens-side or PSF generation mechanisms. It does not mean native optimized optical design.

Each run writes `raw_base_psf_cube.npz`, `psf_cube.npz`, `proxy_transform_manifest.json`, `optical_metrics.json`, `mtf_curves.csv`, and `run_manifest.json`.

All proxy runs must include:

- `encoder_behavior_realized=true`
- `encoder_behavior_realization_level="adapter_proxy"`
- `physical_validation_level="deeplens_base_psf_plus_adapter_proxy"`
- `proxy_transform_applied=true`
- `proxy_transform_name`
