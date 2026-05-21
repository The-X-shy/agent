# Native Optimization Claims

## Claim 1: DeepLens Native Differentiable Component Optimization Is Supported

**Evidence level:** `deeplens_native_component_optimization`

**Required conditions:**
- surface-level probe succeeded
- `requires_grad_true = True`
- `gradient_norm > 0`
- `parameters_changed = True`
- `optimizer_step_executed = True`

**Status:** Supported only for the specific surface/component that passed.

**Caveats:**
- Only validated for the specific surface and objective tested
- Does NOT imply lens-file PSF optimization
- Does NOT imply end-to-end optical-HSI differentiable optimization
- Requires torch autograd infrastructure

---

## Claim 1B: DeepLens Native Differentiable Lens Optimization Is Supported

**Evidence level:** `deeplens_native_lens_optimization`

**Required conditions:**
- lens file loaded
- lens-level PSF/image loss backward succeeds
- `gradient_norm > 0`
- `parameters_changed = True`
- `optimizer_step_executed = True`

**Status:** Depends on lens-file probe outcome.

---

## Claim 2: DeepLens-Backed Black-Box Co-Design Is Supported

**Evidence level:** `deeplens_adapter_proxy` / `codesign_loop`

**Required conditions:**
- Phase 18 evidence exists
- `psf_source = "deeplens_parameterized"`
- `fallback_used = False`
- HSI reconstruction metrics present (PSNR, SAM, ERGAS)
- `differentiable = False` is ALLOWED (black-box)

**Status:** Supported (from Phase 18)

---

## Claim 3: DeepLens Native Optical-HSI Co-Design Is Supported

**Evidence level:** `deeplens_native_optical_hsi_codesign`

**Required conditions:**
- Native differentiable lens optimization success
- DeepLens PSF/image feeds HSI loss
- HSI loss backward reaches an optical parameter
- `optimizer.step()` improves an HSI metric

**Status:** Needs followup.

---

## Explain Claim Fields

The `explain_claim` output includes these native-optimization-specific fields:

| Field | Description |
|---|---|
| `differentiable` | Whether gradients flowed through autograd |
| `native_parameter_update` | Whether optimizer.step changed parameters |
| `gradient_norm` | L2 norm of gradients across all parameters |
| `parameters_changed` | Whether parameter norms differ before/after |
| `loss_before` | Scalar loss before optimization step |
| `loss_after` | Scalar loss after optimization step |
| `lens_class` | Which DeepLens lens class was probed |
| `realization_level` | native / semi_native / adapter_proxy / unavailable |

## Claim Boundary

- **Allowed:** Claims about gradient flow, parameter updates, and loss reduction for specific lens classes
- **Not allowed:** Claims about physical optical performance without real camera validation
- **Not allowed:** Claims about HSI reconstruction improvement without end-to-end native optimization pipeline
