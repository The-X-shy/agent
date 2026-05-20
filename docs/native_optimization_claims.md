# Native Optimization Claims

## Claim 1: DeepLens Native Differentiable Optimization Is Supported

**Evidence level:** `deeplens_native_optimization`

**Required conditions:**
- `differentiable = True`
- `native_parameter_update = True`
- `gradient_norm > 0`
- `parameters_changed = True`
- `fallback_used = False`
- `realization_level = "native"`

**Status:** Depends on probe outcome for each lens class.

**Caveats:**
- Only validated for the specific lens class and objective tested
- Does NOT imply end-to-end optical-HSI differentiable optimization
- Requires torch autograd infrastructure

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

## Claim 3: DeepLens Native Optimization Improves HSI Reconstruction

**Evidence level:** `deeplens_native_optimization` + `hsi_reconstruction`

**Required conditions:**
- Native optimization probe result exists
- HSI reconstruction metrics before AND after native optimization exist
- `final_score > initial_score`
- `native_parameter_update = True`

**Status:** Needs followup — requires integration of native optimization with HSI pipeline (future Phase 20+)

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
