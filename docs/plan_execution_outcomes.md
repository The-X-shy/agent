# Plan Execution Outcomes

Phase 38 separates loop status from evidence outcome.

## Loop Status

- `dry_run_only`: plan was generated but not executed.
- `completed`: local execution path closed with a real outcome, including report-only or structured unsupported outcomes.
- `stopped`: no executable design was available.
- `failed`: an unhandled execution failure prevented a structured outcome.

## Evidence Levels

- `local_execution_completed`: a local scientific execution completed.
- `report_only`: report generation completed and only documents the negative result boundary.
- `structured_unsupported`: the requested local scientific path is unavailable, unsupported, or needs follow-up.

## ClaimGate Rules

- `report_only` can support only that the negative result was documented.
- `structured_unsupported` can support only that a boundary was detected.
- `local_execution_completed` is capped by the backend claim ceiling.

## Memory and State

Memory stores selected, attempted, and skipped designs in entry metadata.

StateStore records:

- `last_executed_design`
- `last_execution_status`
- `last_claim_decision`
- `pending_actions`
- `known_unsupported_claims`
- `known_supported_claims`
