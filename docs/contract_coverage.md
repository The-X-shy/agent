# Contract Coverage Dashboard

The contract coverage dashboard provides a quantitative assessment of system readiness across all contract dimensions.

## Metrics

| Metric | Description |
|---|---|
| handler_contract_coverage | Fraction of handlers with execution contracts |
| design_mapping_coverage | Fraction of designs mapped to handlers |
| remote_contract_coverage | Fraction of handlers with remote contracts |
| artifact_contract_coverage | Fraction of handlers with artifact contracts |
| report_contract_coverage | Fraction of report types with contracts |
| claim_policy_coverage | Fraction of evidence levels with policies |
| test_coverage_proxy | Estimated test coverage per handler |
| doc_coverage_proxy | Estimated doc coverage per handler |
| overall_system_readiness_score | Weighted average (0.0 - 1.0) |

## Usage

```bash
python -m optiresearch.cli export-contract-coverage-dashboard
```

Outputs to `workspace/system_capability/contract_coverage.{json,md}`
