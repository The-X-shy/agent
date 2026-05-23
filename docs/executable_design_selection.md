# Executable Design Selection

`CandidatePlanEvaluator.select_executable_designs()` selects designs by ranked score and execution mode.

## Dry Run

Dry run keeps the top-ranked design for display. No execution is attempted, so designs that need user data can still be shown.

## Local

Local mode accepts only recommendations:

- `execute_now`
- `dry_run_first`

Local mode skips:

- `needs_user_data`
- `needs_remote`
- unsupported backend-task pairs
- non-actionable recommendations such as `defer`

Report-only plans are executable, but they are reserved as a fallback when a scientific local design exists.

## Remote Opt-In

Remote-required designs are eligible only when `allow_remote=true`. The local plan execution command does not start remote work by default.

## Selection Metadata

The selection result records:

- `selected_design`
- `selected_design_rank`
- `skipped_higher_ranked_designs`
- `executable_selection_reason`
- `stop_reason`

If no design is executable, the loop stops with `stop_reason=no_executable_design`.
