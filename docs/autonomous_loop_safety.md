# Autonomous Loop Safety (Phase 25)

Safety mechanisms in the closed-loop autonomous research agent.

## Remote Safety Guards

1. **Default dry_run**: No experiments execute without explicit user choice
2. **Remote opt-in**: `execution_mode=remote_opt_in` only works when:
   - `allow_remote=true` (explicit flag)
   - `remote_worker_id` is set (specific worker)
3. **Command allowlist**: All generated CLI commands pass `validate_remote_command()`
4. **No auto-retry**: Remote failures are not retried more than once
5. **Runtime cap**: `max_runtime_minutes_per_iter` limits per-iteration execution time

## Claim Gate Hard Enforcement

When `strict_claim_gate=true` (default):
- Every proposed claim is checked through `ClaimGateV2`
- Claims judged `unsupported` are rejected:
  - `safe_wording` replaces original inflated claim
  - Original claim is NOT preserved
  - Trajectory report uses safe wording only
- Violation types that trigger rejection:
  - proxy_as_waveoptics, geometric_as_coherent, synthetic_as_real
  - black_box_as_native, unsupported_path_as_supported

## Prohibited Actions

- No automatic git commits
- No automatic code modifications (`allow_code_modification=false` by default)
- No bypass of claim gate (`strict_claim_gate=true` by default)
- No remote execution without explicit opt-in
- No elevation of proxy/geometric/synthetic results to higher claims
