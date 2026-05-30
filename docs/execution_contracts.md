# Execution Contracts

Execution contracts formally link handlers to skills, designs, backends, and define input/output schemas for each execution mode.

## Contract Model

An `ExecutionContract` specifies:

- **handler_id**: The handler this contract applies to
- **skill_id**: The associated skill
- **design_ids**: Compatible design strategies
- **backend_ids**: Allowed optical backends
- **execution_modes**: Supported modes (dry_run, local, remote_opt_in)
- **required_inputs / required_outputs**: Input/output artifacts
- **required_metrics**: Metrics that must be computed
- **evidence_level_mapping**: Evidence level per execution mode
- **claim_ceiling_mapping**: Claim ceiling per execution mode
- **failure_modes**: Known failure scenarios
- **retry_policy / timeout_policy**: Execution policies

## Core Contracts

12 core execution contracts are defined in `tests/test_core_handler_execution_contracts.py`:

1. `ec_deeplens_native_geolens_hsi` - Native GeoLens HSI co-design
2. `ec_stable_native_lens_hsi` - Stable native lens HSI
3. `ec_native_geolens_benchmark` - GeoLens stability benchmark
4. `ec_component_first_probe` - Component-first probe
5. `ec_component_surrogate_hsi` - Component surrogate HSI
6. `ec_trainable_param_inspection` - Trainable parameter inspection
7. `ec_autograd_audit` - Autograd audit
8. `ec_curriculum_probe` - Curriculum learning probe
9. `ec_regularized_probe` - Regularized probe
10. `ec_objective_redesign` - Objective redesign
11. `ec_param_reduction` - Parameter reduction sweep
12. `ec_report_negative_result` - Negative result reporting

## Validation

```bash
python -m optiresearch.cli validate-execution-contracts
```

Checks:
- Each handler has a contract
- Design/Skill/Backend references are valid
- evidence_level does not exceed claim_ceiling
- Required outputs and execution modes are specified
