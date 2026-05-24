# Remote Handler Claim Ceiling

Phase 44 keeps local and remote claim ceilings separate.

## Local Ceiling

`remote_native_geolens_validation` cannot run locally.

When the handler is evaluated with `execution_target=local`, the final ceiling is:

```text
needs_followup
```

## Remote Ceiling

When the handler runs on WSL and `remote_validation_passed=true`, the handler ceiling is:

```text
native_lens_simulation
```

The remote result still cannot support:

- coherent wave-optics claims;
- real HSI performance claims;
- claims based on proxy fallback;
- claims based on missing artifacts or missing result fields.

If remote validation fails, `ClaimCeilingResolver` returns `needs_followup`.
