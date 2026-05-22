# Real LLM Validation (Phase 27)

End-to-end validation of the LLM-assisted autonomous research loop using
real DeepSeek provider.

## Prerequisites

```bash
export DEEPSEEK_API_KEY="<your-key>"
export DEEPSEEK_BASE_URL="https://api.deepseek.com"  # default
export DEEPSEEK_MODEL="deepseek-v4-pro"               # default
```

## Validation Steps

### 1. Provider Environment Check

```bash
python -m optiresearch.cli check-llm-provider --provider deepseek
```

Expected: `status=available`, model and base_url populated, no API key in output.

### 2. Planner Smoke Test

```bash
OPTIRESEARCH_ENABLE_REAL_LLM_TESTS=1 \
DEEPSEEK_API_KEY=$DEEPSEEK_API_KEY \
python -m pytest tests/test_real_deepseek_planner.py -v
```

Validates: proposals generated, schema validation passes, claim gate applied, trace saved without secrets.

### 3. CLI Planner Validation

```bash
python -m optiresearch.cli plan-with-llm \
  --objective "investigate differentiable wave-optics alternatives without overclaiming" \
  --provider deepseek
```

Expected output: planner_run_id, status, proposals_count, selected_proposal, validation_errors, claim_gate_decision, fallback_used.

### 4. Autonomous Loop Dry Run

```bash
python -m optiresearch.cli run-autonomous-research-loop-v2 \
  --objective "investigate differentiable wave-optics alternatives without overclaiming" \
  --max-iterations 2 \
  --execution-mode dry_run \
  --planner-mode llm_first_with_rule_fallback \
  --llm-provider deepseek
```

Dry run only — no experiments executed. LLM generates proposals, ClaimGate and PlannerValidator execute. On DeepSeek failure, falls back to StrategyEngine.

### 5. Local Minimal Execution

```bash
python -m optiresearch.cli run-autonomous-research-loop-v2 \
  --objective "improve native lens simulation HSI co-design stability" \
  --max-iterations 1 \
  --execution-mode local \
  --planner-mode llm_first_with_rule_fallback \
  --llm-provider deepseek
```

Single iteration, local only, no remote tasks. ClaimGate result enters loop_result. MemoryV2 updated.

### 6. Validation Report

```bash
python -m optiresearch.cli export-llm-provider-validation-report \
  --planner-run-id <planner_run_id> \
  --loop-id <loop_id>
```

Consolidated report at `workspace/reports/llm_provider_validation_report.md`.

## Phase 28: Multi-Iteration Local Execution

### 7. Multi-Iteration Local Loop

```bash
python -m optiresearch.cli run-autonomous-research-loop-v2 \
  --objective "execute lightweight stable native lens HSI co-design" \
  --max-iterations 2 \
  --execution-mode local \
  --planner-mode llm_first_with_rule_fallback \
  --llm-provider deepseek \
  --prefer-executable-actions \
  --allowed-backends "phase_to_fft_proxy" \
  --allowed-task-types "stable_lens_hsi_codesign,lightweight_psf_probe"
```

Multi-iteration loop with feedback context. LLM receives previous iteration metrics to inform next strategy. Lightweight experiments run via pure-PyTorch FFT without DeepLens.

### 8. Real DeepSeek Multi-Iteration Test

```bash
OPTIRESEARCH_ENABLE_REAL_LLM_TESTS=1 \
DEEPSEEK_API_KEY=$DEEPSEEK_API_KEY \
python -m pytest tests/test_real_deepseek_local_loop_execution.py -v
```

## Safety Constraints

- No API key in any trace file
- No shell commands in LLM output
- No forbidden actions (git, rm, sudo, pip, curl, wget, ssh)
- Claim gate hard enforcement with safe wording
- Dry run by default; remote requires explicit opt-in
- Fallback to rule-based StrategyEngine on any LLM failure

## Known Limitations

- DeepSeek API key required for real LLM validation
- Network errors or rate limits may trigger fallback
- LLM may produce malformed JSON (caught by parser + validator)
- Remote execution requires explicit `--allow-remote` flag
