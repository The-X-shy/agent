# Autonomous Research Loop v2 (Phase 25)

The closed-loop autonomous research agent composes Phase 24 components
into an automated experiment cycle: strategy → plan → execute → diagnose →
claim gate → memory → decide.

## Architecture

```
AutonomousLoopSpec
    |
    v
run_autonomous_research_loop()         # runtime/autonomous_research_loop.py
    |
    +--> StrategyEngine.recommend()    # agents/strategy_engine.py
    +--> compile_experiment_spec()     # agents/strategy_to_spec.py
    +--> ExperimentControllerV2.run()  # runtime/experiment_controller_v2.py
    +--> audit_autograd_graph()        # diagnostics/autograd_auditor.py
    +--> ClaimGateV2.check_claim()     # memory/claim_gate_v2.py
    +--> ResearchMemoryV2.add_entry()  # memory/research_memory_v2.py
    +--> evaluate_trajectory()         # agents/trajectory_evaluator.py
```

## Execution Modes

- **dry_run** (default): Output strategy + proposed CLI commands. No experiments run.
- **local**: Full iteration loop with local experiment execution.
- **remote_opt_in**: Only when `allow_remote=true` AND `remote_worker_id` set.

## CLI

```bash
# Dry run (default, safe)
python -m optiresearch.cli run-autonomous-research-loop-v2 \
  --objective "test co-design stability" \
  --max-iterations 3 \
  --execution-mode dry_run

# Local execution
python -m optiresearch.cli run-autonomous-research-loop-v2 \
  --objective "improve HSI reconstruction" \
  --max-iterations 2 \
  --execution-mode local

# Export trajectory report
python -m optiresearch.cli export-autonomous-loop-v2-report \
  --loop-id aloop2_xxxxxxxxxxxxxxxx
```

## Output Structure

```
workspace/autonomous_loops_v2/<loop_id>/
├── loop_spec.json
├── loop_result.json
├── iteration_001/
│   ├── 01_strategy.json
│   ├── 02_spec.json
│   ├── 03_execution.json
│   ├── 04_autograd.json    (if differentiable path)
│   ├── 05_claim_gate.json
│   └── 06_memory.json
└── autonomous_research_loop_report.md
```

## Safety Rules

1. Default execution_mode is `dry_run`
2. Remote execution requires explicit opt-in (`allow_remote=true`)
3. All remote commands must pass the command allowlist
4. Strict claim gate (`strict_claim_gate=true`) uses safe_wording for unsupported claims
5. No automatic git commits or code modifications
6. Remote failure → no auto-retry beyond 1 attempt

## Programmatic API

```python
from optiresearch.schemas.autonomous_loop import AutonomousLoopSpec
from optiresearch.runtime.autonomous_research_loop import run_autonomous_research_loop

spec = AutonomousLoopSpec(
    objective="improve HSI reconstruction stability",
    max_iterations=3,
    execution_mode="dry_run",
)
result = run_autonomous_research_loop(spec)
print(f"Status: {result.status}")
print(f"Iterations: {len(result.iterations)}")
```
