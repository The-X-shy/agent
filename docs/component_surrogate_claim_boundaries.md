# Component Surrogate Claim Boundaries

Component surrogate HSI co-design supports a narrow claim scope.

## Supported Claims

| Claim | Status |
|---|---|
| Component surrogate PSF requires grad | Supported |
| Synthetic HSI reconstruction loss requires grad | Supported |
| Component parameters receive gradients from the HSI loop | Supported |
| Component parameters changed after optimizer step | Supported |
| Synthetic reconstruction metrics changed during surrogate optimization | Supported |

## Blocked Claims

| Claim | Why blocked |
|---|---|
| Full GeoLens lens-level optimization | Surrogate PSF is not full GeoLens |
| Native physical lens optimization | No physical lens backend is used |
| Real HSI performance | Dataset is synthetic |
| Real camera validation | No camera measurement is used |
| Full wave-optics co-design | Surrogate is not full wave propagation |

## Claim Ceiling

The maximum claim is `component_surrogate_hsi_codesign`.
`ClaimGateV2` rejects promotion to full GeoLens, native physical lens, real HSI,
real camera, or full wave-optics claims.
