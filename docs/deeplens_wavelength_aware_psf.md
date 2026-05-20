# DeepLens Wavelength-Aware PSF Contract

Phase 12 makes the HSI PSF contract explicit for DeepLens outputs.

The adapter writes these fields into metrics and manifests:

- `wavelength_aware_psf`
- `wavelengths_nm`
- `wavelength_count`
- `psf_band_axis`
- `depth_count`
- `psf_cube_shape`
- `wavelength_sampling_method`
- `hsi_forward_compatible`
- `native_wavelength_physics`

`wavelength_aware_psf=true` means the exported PSF cube has a wavelength axis shaped `[depth, wavelength, height, width]`.

For adapter-proxy runs, `native_wavelength_physics=false`. This must not be written as native physical validation.

New capability names:

- `wavelength_aware_psf_export_available`
- `native_wavelength_physics_available`
- `hsi_forward_compatible_psf_available`

## Phase 13 Status

The wavelength-aware PSF contract is included in the final benchmark registry (Group B) and evidence distribution. The contract is interface-level unless native wavelength physics validation is performed (Phase 14). See `docs/claim_boundary.md` for qualified claims.

