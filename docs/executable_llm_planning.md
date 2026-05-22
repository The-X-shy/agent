# Executable LLM Planning (Phase 28)

Guides the LLM planner to prefer executable actions over `stop_and_report`.

## Mode Activation

Set `prefer_executable_actions=True` in `AutonomousLoopSpec` or use CLI flag:

```bash
python -m optiresearch.cli run-autonomous-research-loop-v2 \
  --prefer-executable-actions ...
```

## Prompt Changes

When `prefer_executable_actions` is True, the LLM planner prompt includes:

> EXECUTABLE ACTIONS REQUIRED: You are running in executable mode. Prefer executable actions (retry_with_smaller_lr, enable_rollback, run_ablation, probe_waveoptics_path) over stop_and_report. Only select stop_and_report if ALL executable actions are blocked by safety constraints or claim ceiling violations.

## Rejected-Proposal Scanning

If the LLM still selects `stop_and_report` with `prefer_executable_actions`:

1. The planner scans rejected proposals for an executable alternative
2. If found, the alternative replaces `stop_and_report` as the selected proposal
3. If none found, the planner falls back to `StrategyEngine` for a rule-based executable strategy

## Autonomous Loop Fallback

In the autonomous loop, if `prefer_executable_actions` is True and the LLM returns `stop_and_report` in a non-final iteration:

1. The loop ignores the LLM's stop_and_report
2. Falls back to `StrategyEngine.recommend()` (rule-based)
3. Marks `metadata.planner = "fallback"` with `fallback_reason = "prefer_executable_with_llm_stop"`

This ensures at least one experiment executes before the loop stops.

## Allowed Executable Actions

| Action | Task Type | Description |
|--------|-----------|-------------|
| `retry_with_smaller_lr` | `stable_lens_hsi_codesign` | Reduce optical LR to 1e-6 |
| `enable_rollback` | `stable_lens_hsi_codesign` | Enable rollback protection |
| `run_ablation` | `stable_lens_hsi_codesign` | Systematic ablation study |
| `probe_waveoptics_path` | `lightweight_psf_probe` | FFT-based PSF probe |
