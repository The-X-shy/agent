# LLM Planner

LLM-assisted autonomous research planner for differentiable optics.

## Architecture

```
LLMPlanner.plan()
  |
  ├── build_context() → BackendRegistry + ResearchMemoryV2
  ├── _get_provider() → LLM Provider Registry
  ├── _call_provider()
  │     ├── mock → build_mock_proposals()
  │     └── deepseek → build_planner_prompt() + DeepSeekProvider.complete()
  ├── _parse_proposals() → LLMPlannerProposal[]
  ├── _validate_all() → PlannerValidator (10 checks)
  ├── rank_proposals() → sorted by risk_level
  ├── _apply_claim_gate() → ClaimGateV2.check_claim()
  ├── _record_trace() → workspace/planner_traces/
  └── (on failure) _fallback_result() → StrategyEngine
```

## Components

### LLMPlannerProposal

Each proposal contains:
- `proposal_id` — unique identifier
- `hypothesis` — research hypothesis
- `rationale` — reasoning behind the proposal
- `recommended_action` — action to take (retry_with_smaller_lr, run_ablation, stop_and_report, etc.)
- `backend_id` — target optical backend
- `task_type` — experiment task type
- `proposed_claim` — scientific claim being made
- `safe_wording` — claim gate safe alternative (populated if original claim was downgraded)
- `risk_level` — low/medium/high

### Provider Selection

The planner supports multiple LLM providers:
- **mock** — deterministic proposals for testing (always available)
- **deepseek** — real DeepSeek API (requires `DEEPSEEK_API_KEY`)

### Validation Pipeline

1. **Schema validation** — proposal_id, recommended_action, risk_level
2. **Backend validation** — backend must exist in allowed list
3. **Task type validation** — task must be supported
4. **Claim ceiling** — backend must support the claim level
5. **Execution mode** — remote requires explicit opt-in
6. **Shell command check** — no shell patterns in claim/rationale/hypothesis
7. **Forbidden actions** — no git/rm/sudo/pip/curl/wget/ssh keywords
8. **Dataset claim** — synthetic backends cannot claim "real" or "physical"
9. **Wave-optics claim** — geometric/proxy backends cannot claim "coherent" or "full wave"
10. **All rejected** — if no valid proposals, fallback to StrategyEngine

### Claim Gate

After selecting the best proposal, `ClaimGateV2.check_claim()` is applied:
- **supported** — claim is within backend capability
- **qualified** — claim needs caveats
- **needs_followup** — more evidence required
- **unsupported** — claim exceeds backend capability; `safe_wording` replaces original

### Fallback

On any failure (provider error, parse error, validation failure, all proposals rejected), the planner falls back to `StrategyEngine.recommend()` which uses 8 rule-based strategies.

## CLI

```bash
# Generate proposals
python -m optiresearch.cli plan-with-llm \
  --objective "investigate differentiable wave-optics alternatives" \
  --provider deepseek \
  --execution-mode dry_run

# List saved traces
python -m optiresearch.cli list-planner-traces

# Inspect a trace
python -m optiresearch.cli inspect-planner-trace --planner-run-id <id>

# Export report
python -m optiresearch.cli export-llm-planner-report --planner-run-id <id>
```
