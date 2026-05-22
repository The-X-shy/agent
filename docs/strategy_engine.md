# Strategy Engine

`StrategyEngine` automatically recommends the next experimental action based
on result metrics, backend capabilities, and known failure patterns from
Phases 18-23.

## Built-in Rules (Priority Order)

| # | Rule | Trigger | Action |
|---|------|---------|--------|
| 1 | Large gradient | optical_gradient_norm > 100 | Reduce LR 100x + rollback |
| 2 | High rollback ratio | rollback_count / steps > 0.5 | Freeze optics / ablation |
| 3 | Zero gradient | grad_norm == 0 after optimizer step | Audit autograd |
| 4 | Loss increase | loss_after > loss_before without rollback | Enable rollback |
| 5 | Claim downgraded | claim_downgraded == True | Reword claim |
| 6 | Recon loss increase | recon_loss_after > recon_loss_before | More warmup steps |
| 7 | Max gradient spike | grad_max > 10 | Clip + reduce LR |
| 8 | PSF energy drift | psf_energy_delta > 0.5 | Increase PSF reg |

## Risk Levels

- **low**: Reversible change, low blast radius (e.g., reduce LR)
- **medium**: May change claim scope (e.g., freeze optics)
- **high**: Indicates possible fundamental issue (e.g., autograd break)

## Default Recommendations

When no specific rule fires:
- **Stable training succeeded** → run remote validation
- **Loss decreased** → run remote validation
- **Otherwise** → stop and report

## CLI

```bash
python -m optiresearch.cli recommend-next-strategy \
  --backend-id deeplens_geolens_geometric \
  --latest-result-json '{"optical_gradient_norm": 500}'
```

## Programmatic API

```python
from optiresearch.agents.strategy_engine import StrategyEngine

engine = StrategyEngine()
rec = engine.recommend(
    {"optical_gradient_norm": 500, "max_steps": 10},
    "deeplens_geolens_geometric",
)
print(rec.recommended_action)  # "retry_with_smaller_lr"
print(rec.proposed_cli_commands)
```

## Adding Custom Rules

Rules are simple condition-action pairs. To add a custom rule:

```python
engine = StrategyEngine()
engine._rules.insert(0, {
    "id": "my_custom_rule",
    "condition": lambda r: r.get("my_metric", 0) > 100,
    "action": "my_custom_action",
    "rationale": "My custom rationale...",
    "risk_level": "low",
    "proposed_commands": ["my-custom-cli-command"],
    "required_evidence": ["Evidence needed..."],
})
```
