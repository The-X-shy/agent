# Agent Plan Local Execution

Phase 38 turns the Phase 37 dry-run plan into a local execution loop.

## Scope

- Generates strategies, designs, and ranked plan scores from the seed failure.
- Selects the highest-ranked local executable scientific design.
- Skips designs that need user data, remote execution, or unsupported backend-task pairs.
- Attempts the selected design locally.
- If the selected scientific design is unavailable or unsupported, attempts the report-only fallback within the local attempt limit.
- Runs ClaimGate on the final local outcome.
- Records attempted designs, skipped higher-ranked designs, Memory, StateStore, EventBus, and report output.

## Local Execution Result

The loop writes `workspace/agent_plan_executions/<execution_id>/execution_result.json`.

Important fields:

- `selected_design`
- `selected_design_rank`
- `skipped_higher_ranked_designs`
- `attempted_designs`
- `execution_result.status`
- `execution_result.evidence_level`
- `claim_gate_decision`
- `memory_updated`
- `state_snapshots_count`
- `event_count`
- `fallback_to_report_only`

## Safety

Local mode does not dispatch remote work. Remote-capable designs are skipped unless the mode is `remote_opt_in` and remote execution is explicitly allowed.

Report-only fallback records the negative result boundary. It does not upgrade optical improvement claims.
