# Agent Plan Execution Loop

The agent plan execution loop converts a classified failure into local execution evidence.

Pipeline:

1. Classify the seed result.
2. Generate recovery strategies.
3. Generate experiment design candidates.
4. Score candidates.
5. Select executable designs for the requested mode.
6. Execute locally or produce a dry run.
7. Run ClaimGate on the final outcome.
8. Record Memory, StateStore snapshot, EventBus events, and a report.

## Commands

```bash
python -m optiresearch.cli run-agent-plan-execution \
  --objective "recover from native GeoLens optical update instability" \
  --seed-result-path workspace/native_geolens_stabilization/geolens_stabilization_1779550632/sweep_results.json \
  --mode local \
  --execute-top-k 1

python -m optiresearch.cli export-agent-plan-execution-report \
  --execution-id <execution_id>
```

## Local Fallback

If the selected scientific design returns `structured_unsupported`, the loop attempts `report_negative_result_doc` as the low-risk local fallback. This closes the loop without upgrading scientific claims.

## Outputs

- `execution_result.json`
- `events.json`
- `plan_execution_report.md`
- StateStore snapshot under `workspace/agent_state/snapshots/`
