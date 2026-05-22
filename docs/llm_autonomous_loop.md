# LLM-Autonomous Loop

LLM-assisted autonomous research loop for differentiable optics.

## Planner Modes

| Mode | Behavior |
|------|----------|
| `rule_based` | StrategyEngine only (no LLM) |
| `llm_assisted` | LLM first, StrategyEngine fallback |
| `llm_first_with_rule_fallback` | Same as llm_assisted |

All modes fall back to StrategyEngine on LLM failure. The naming distinction preserves semantic intent.

## Iteration Flow

```
for iteration in 1..max_iterations:
  1. Collect previous results and memory
  2. Run planner:
     - LLMPlanner.plan() (if llm_assisted or llm_first_with_rule_fallback)
     - StrategyEngine.recommend() (if rule_based or LLM fallback)
  3. Compile ExperimentSpecV2 from strategy
  4. Execute experiment (or skip in dry_run)
  5. Run ClaimGateV2 on results
  6. Update ResearchMemoryV2
  7. Evaluate trajectory → continue or stop
```

## Execution Modes

| Mode | Experiments | Remote | Use Case |
|------|-------------|--------|----------|
| `dry_run` | None | No | Validation, planning only |
| `local` | Local only | No | Development, testing |
| `remote_opt_in` | Remote opt-in | Requires `--allow-remote` | Production |

## CLI

```bash
# Dry run with LLM planner
python -m optiresearch.cli run-autonomous-research-loop-v2 \
  --objective "investigate differentiable wave-optics alternatives" \
  --max-iterations 3 \
  --execution-mode dry_run \
  --planner-mode llm_first_with_rule_fallback \
  --llm-provider deepseek

# Local execution with rule-based planner
python -m optiresearch.cli run-autonomous-research-loop-v2 \
  --objective "improve native lens simulation stability" \
  --max-iterations 1 \
  --execution-mode local \
  --planner-mode rule_based

# Export report
python -m optiresearch.cli export-autonomous-loop-v2-report --loop-id <id>
```

## Output

Each loop produces:
- `workspace/autonomous_loops_v2/<loop_id>/loop_spec.json` — input specification
- `workspace/autonomous_loops_v2/<loop_id>/loop_result.json` — full result with iterations
- `workspace/autonomous_loops_v2/<loop_id>/trajectory_report.md` — human-readable report

## Safety

- Default `dry_run` — no experiments without explicit choice
- `strict_claim_gate=true` by default
- LLM output validated through PlannerValidator (10 checks)
- ClaimGateV2 applied to every proposed claim
- Safe wording replaces inflated claims
- API keys redacted from all traces
- Remote execution requires explicit `--allow-remote`
