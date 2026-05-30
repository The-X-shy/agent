# Artifact Contracts

Artifact contracts specify required and optional output artifacts for each handler and job type.

## Contract Fields

| Field | Description |
|---|---|
| contract_id | Unique contract identifier |
| handler_id | Associated handler |
| required_artifacts | Files that must exist |
| optional_artifacts | Files that may exist |
| artifact_roles | Role mapping (execution_result, primary_metric, report, etc.) |
| sha256_required | Whether content hashing is required |
| artifactstore_registration_required | Whether ArtifactStore registration is needed |
| evidence_binding_required | Whether ClaimEvidence binding is needed |
| missing_artifact_policy | needs_followup, partial_evidence, or structured_warning |

## Core Contracts

7 core artifact contracts:

1. `ac_diagnostic` - Diagnostic job output
2. `ac_component_probe` - Component probe output
3. `ac_native_geolens_stability` - GeoLens stability run
4. `ac_native_geolens_benchmark` - GeoLens benchmark
5. `ac_benchmark_failure_analysis` - Failure analysis
6. `ac_remote_job` - Generic remote job
7. `ac_agent_plan` - Agent plan execution

## Validation

```bash
python -m optiresearch.cli validate-artifact-contract --run-dir <path> --contract-id <id>
```
