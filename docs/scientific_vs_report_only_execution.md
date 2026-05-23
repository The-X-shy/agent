# Scientific vs Report-Only Execution

Phase 39 adds `lightweight_scientific_execution` as a middle ground between "no local execution possible" and "full native DeepLens execution."

## Execution Paths

| Path | Evidence Level | Metrics? | Requires DeepLens? | Claim Support |
|---|---|---|---|---|
| Report-only fallback | `report_only` | No | No | Document negative result only |
| Structured unsupported | `structured_unsupported` | No | No | Document boundary only |
| Lightweight scientific | `lightweight_scientific_execution` | **Yes** | **No** | Synthetic metric experiments |
| Native DeepLens | `local_execution_completed` | Yes | Yes | Native lens simulation |
| Native wave-optics | `native_waveoptics` | Yes | Yes (coherent ASM) | Wave-optics simulation |

## Selection Priority

In local mode, the `CandidatePlanEvaluator` prefers:

1. **Executable scientific design** (e.g., `objective_redesign_simpler_metric_mse_only`)
2. **Executable backend probe** (e.g., `backend_switch_waveoptics_coherent` — but this returns `needs_followup`)
3. **Report-only fallback** (always deferred to last)
4. **No executable design** (stops with error)

This means if a scientific design is available and executable, it will be attempted before falling back to report-only.

## When Scientific Execution Is Used

Scientific execution is selected when:
- The agent generates a design matching `_is_lightweight_scientific_design()` criteria
- The design has `design_id == "objective_redesign_simpler_metric_mse_only"` or MSE-only loss weights
- The design is ranked higher than report-only designs
- The design is considered locally supported

## When Report-Only Is Used

Report-only is used as a fallback when:
- All scientific designs fail
- No scientific designs are available
- The selected design needs user data or remote execution

## Comparison

### Report-Only Execution
- Produces: `{"report_generated": true}` in metrics
- Evidence: `report_only`
- ClaimGate: Only allows "negative result documented" claims
- Value: Records a boundary was hit

### Scientific Execution
- Produces: Real loss values, MSE, PSNR, improvement detection
- Evidence: `lightweight_scientific_execution`
- ClaimGate: Allows "synthetic experiment shows improvement" claims
- Value: Produces measurable evidence without DeepLens dependency

## Example Output

```json
{
  "execution_result": {
    "status": "completed",
    "evidence_level": "lightweight_scientific_execution",
    "metrics": {
      "reconstruction_loss_before": 0.1234,
      "reconstruction_loss_after": 0.0567,
      "mse_before": 0.1234,
      "mse_after": 0.0567,
      "psnr_before": 9.09,
      "psnr_after": 12.46,
      "improvement_detected": true,
      "metrics_valid": true
    }
  },
  "fallback_to_report_only": false,
  "claim_gate_decision": {
    "decision": "supported",
    "max_allowed_claim": "lightweight_scientific_execution"
  }
}
```
