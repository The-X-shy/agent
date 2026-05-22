# LLM Planner Safety

Safety constraints for LLM-assisted research planning.

## System Prompt Constraints

The LLM planner prompt enforces 10 hard constraints:

1. Only propose actions from the allowed action set
2. Only use backends from the provided backend registry
3. Only propose task types from the allowed list
4. Never propose claims exceeding the backend claim ceiling
5. Never output shell commands or executable code
6. Never propose forbidden actions (git, rm, sudo, pip, curl, wget, ssh)
7. Never claim real/experimental results from synthetic data
8. Never claim coherent wave-optics from geometric models
9. Always provide safe_wording for every proposed_claim
10. Proposals must be valid JSON matching the schema

## PlannerValidator Checks

Located in `optiresearch/agents/planner_validator.py`:

1. Schema validation (proposal_id, recommended_action, risk_level)
2. Backend exists in allowed list
3. Task type is supported
4. Claim ceiling check (backend capability vs claim)
5. Execution mode check (remote requires opt-in)
6. No shell patterns in proposed_claim (`;`, `&&`, `||`, `|`, backticks, `$()`)
7. No shell patterns in rationale
8. No shell patterns in hypothesis
9. No forbidden keywords (git, rm, sudo, pip, curl, wget, ssh, scp, chmod, chown, kill, reboot, shutdown, docker, systemctl)
10. No forbidden keywords in rationale

Additional content checks:
- `validate_dataset_claim()` — synthetic backends cannot claim "real" or "physical"
- `validate_waveoptics_claim()` — geometric/proxy backends cannot claim "full wave" or "coherent"

## ClaimGateV2 Violations

8 violation types that trigger claim downgrade or rejection:

| Violation | Trigger | Decision |
|-----------|---------|----------|
| proxy_as_waveoptics | phase_to_fft_proxy claims waveoptics | unsupported |
| geometric_as_coherent | geometric PSF claims coherent | unsupported |
| synthetic_as_real | synthetic data claims real | unsupported |
| differentiable_as_improves | differentiable=true but no loss decrease | qualified |
| local_only_as_robust | local-only execution claims robustness | qualified |
| rollback_protection_as_improvement | rollback active but no accepted updates | qualified |
| unsupported_path_as_supported | coherent ASM with requires_grad=False | unsupported |
| black_box_as_native | black-box backend claims native gradient | unsupported |

## Fallback Surface

On any failure in the LLM path:
- Provider unavailable → StrategyEngine fallback
- HTTP error / timeout → StrategyEngine fallback
- Parse error (non-JSON) → StrategyEngine fallback
- All proposals rejected by validator → StrategyEngine fallback
- Claim gate downgrades all proposals → safe_wording applied

The StrategyEngine uses only hardcoded rules and never makes external API calls.

## Trace Audit

All planner traces are sanitized before writing to disk:
- API keys redacted
- Authorization headers stripped
- Environment variable values replaced with `[REDACTED]`

Traces are saved to `workspace/planner_traces/<run_id>/` for auditability.
