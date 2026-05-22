# LLM Multi-Iteration Execution (Phase 28)

Multi-iteration autonomous research loop with LLM planning and feedback.

## Architecture

```
Iteration 1:
  LLMPlanner.plan(prefer_executable=True)
    → strategy → compile ExperimentSpecV2 → execute → feedback context

Iteration 2:
  LLMPlanner.plan(prefer_executable=True, recent_results=[feedback_1])
    → strategy → compile → execute → feedback context

...

TrajectoryEvaluator → best_iteration → loop_result
```

## Feedback Context

Between iterations, `build_feedback_context()` extracts scalar metrics from the previous iteration:

- `previous_action`, `previous_task_type`, `previous_backend`
- `loss_before`, `loss_after`, `improvement_detected`
- `rollback_count`, `accepted_update_count`
- `claim_gate_decision`, `failure_mode`

This context is passed as `recent_results` to the LLM planner for the next iteration, enabling the LLM to adapt its strategy based on actual results.

## Execution Flow

1. **Strategy**: LLM planner (or rule-based fallback) generates recommendation
2. **Plan**: Strategy compiled to `ExperimentSpecV2` via `strategy_to_spec`
3. **Execute**: `ExperimentControllerV2.run_local()` dispatches to lightweight or DeepLens runtime
4. **Diagnose**: Autograd audit (extension point)
5. **Claim Gate**: `ClaimGateV2.check_claim()` validates claim
6. **Memory**: `ResearchMemoryV2` appended with experiment outcome
7. **Feedback**: Context extracted for next iteration
8. **Decide**: `TrajectoryEvaluator` determines continue/stop

## Best Result Tracking

The `TrajectoryEvaluator` identifies the best iteration (lowest reconstruction loss). This is stored in `loop_result.best_result`.

## Fallback Behavior

| Scenario | Action |
|----------|--------|
| LLM returns stop_and_report (non-final) | Rule-based executable strategy |
| LLM returns invalid proposal | PlannerValidator rejects → fallback |
| All proposals rejected | StrategyEngine fallback |
| Provider error | StrategyEngine fallback |

## CLI

```bash
python -m optiresearch.cli run-autonomous-research-loop-v2 \
  --objective "execute lightweight stable native lens HSI co-design" \
  --max-iterations 2 \
  --execution-mode local \
  --planner-mode llm_first_with_rule_fallback \
  --llm-provider deepseek \
  --prefer-executable-actions \
  --max-runtime-minutes-per-iter 2 \
  --allowed-backends "phase_to_fft_proxy" \
  --allowed-task-types "stable_lens_hsi_codesign,lightweight_psf_probe"
```
